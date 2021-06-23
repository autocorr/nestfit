#!/usr/bin/env python3

import os
# NOTE This is a hack to avoid pecular file locking issues for the
# externally linked chunk files on the NRAO's `lustre` filesystem.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import time
import shutil
import inspect
import warnings
import itertools
import multiprocessing
from pathlib import Path
from collections import Iterable

import h5py
import numpy as np
import scipy as sp

import spectral_cube
from astropy import (convolution, units)
from astropy.io import fits

from nestfit.models import MODELS
from nestfit.synth_spectra import get_test_spectra
from nestfit.core.core import (Dumper, run_multinest)
from nestfit.prior_constructors import get_irdc_priors
from nestfit.models.ammonia import (
        amm_predict, AmmoniaSpectrum, AmmoniaRunner,
        test_profile_predict,
)


def nans(shape, dtype=None):
    return np.full(shape, np.nan, dtype=dtype)


class NoiseMap:
    def __init__(self, data):
        # NOTE The axes in the data cube are transposed, so these need to
        # be as well
        self.data = data.transpose()
        self.shape = self.data.shape

    @classmethod
    def from_pbimg(cls, rms, pb_img):
        shape = pb_img.shape
        naxes = len(shape)
        if naxes == 4:
            pb_img = pb_img[0,0]
        elif naxes == 3:
            pb_img = pb_img[0]
        elif naxes == 2:
            pass
        else:
            raise ValueError(f'Cannot parse shape : {shape}')
        # A typical primary beam image will be masked with NaNs, so replace
        # them in the noise map with Inf values.
        img = rms / pb_img
        img[~np.isfinite(img)] = np.inf
        return cls(img)

    def get_noise(self, i_lon, i_lat):
        return self.data[i_lon, i_lat]


class NoiseMapUniform:
    def __init__(self, rms):
        self.rms = rms
        self.shape = None

    def get_noise(self, i_lon, i_lat):
        return self.rms


class DataCube:
    def __init__(self, cube, noise_map, trans_id=None):
        if isinstance(noise_map, (float, int)):
            self.noise_map = NoiseMapUniform(noise_map)
        else:
            self.noise_map = noise_map
        self.trans_id = trans_id
        self._header = cube.header.copy()
        self.dv = self.get_chan_width(cube)
        self.data, self.xarr = self.data_from_cube(cube)
        self.varr = self.velo_axis_from_cube(cube)
        self.shape = self.data.shape
        # NOTE data is transposed so (s, b, l) -> (l, b, s)
        self.spatial_shape = (self.shape[0], self.shape[1])
        self.nchan = self.shape[2]
        if self.noise_map.shape is not None:
            assert self.spatial_shape == self.noise_map.shape

    @property
    def full_header(self):
        return self._header

    @property
    def simple_header(self):
        # FIXME first two axes must be angular coordinates
        keys = (
                'SIMPLE', 'BITPIX',
                'NAXIS',
                'NAXIS1', 'NAXIS2',
                'WCSAXES',
                'CRPIX1', 'CRPIX2',
                'CDELT1', 'CDELT2',
                'CUNIT1', 'CUNIT2',
                'CTYPE1', 'CTYPE2',
                'CRVAL1', 'CRVAL2',
                'RADESYS',
                'EQUINOX',
        )
        hdict = {}
        for k in keys:
            try:
                hdict[k] = self._header[k]
            except KeyError:
                continue
        hdict['NAXIS'] = 2
        hdict['WCSAXES'] = 2
        coord_sys = ('ra', 'dec', 'lon', 'lat')
        # CTYPE's of form "RA---SIN"
        assert hdict['CTYPE1'].split('-')[0].lower() in coord_sys
        assert hdict['CTYPE2'].split('-')[0].lower() in coord_sys
        return hdict

    def get_chan_width(self, cube):
        axis = cube.with_spectral_unit(
                'km/s', velocity_convention='radio').spectral_axis
        return abs(axis[1] - axis[0]).value

    def data_from_cube(self, cube):
        # convert intensity units to Kelvin
        if cube.unit == '':
            print('-- Assuming cube intensity units of K')
            cube._unit = units.K
        elif cube.unit != 'K':
            cube = cube.to('K')
        # convert frequency axis to units of Hz
        if cube.spectral_axis.unit != 'Hz':
            cube = cube.with_spectral_unit('Hz')
        axis = cube.spectral_axis.value.copy()
        nu_chan = axis[1] - axis[0]
        # ensure that the frequency axis is in ascending order
        if nu_chan < 0:
            cube = cube[::-1]
            axis = cube.spectral_axis.value.copy()
        # data is transposed such that the frequency axis is contiguous (now
        # the last or right-most in of the indices)
        data = cube._data.transpose().copy()
        return data, axis

    def velo_axis_from_cube(self, cube):
        varr = (
                cube.with_spectral_unit('km/s', velocity_convention='radio')
                .spectral_axis.value
        )
        # The frequency array `xarr` will be in units of Hertz and in ascending order,
        # thus in order for the axes to match element-wise, the velocity axis must be
        # in descending order.
        if varr[1] > varr[0]:
            return varr[::-1].copy()
        else:
            return varr.copy()

    def get_spec_data(self, i_lon, i_lat):
        arr = self.data[i_lon,i_lat,:]  # axes reversed from typical cube
        noise = self.noise_map.get_noise(i_lon, i_lat)
        has_nans = np.isnan(arr).any() or np.isnan(noise)
        return self.xarr, arr, noise, self.trans_id, has_nans


class CubeStack:
    def __init__(self, cubes):
        assert isinstance(cubes, Iterable)
        self.cubes = cubes
        self.n_cubes = len(cubes)

    def __iter__(self):
        for cube in self.cubes:
            yield cube

    @property
    def full_header(self):
        return self.cubes[0].full_header

    @property
    def simple_header(self):
        return self.cubes[0].simple_header

    @property
    def shape(self):
        return self.cubes[0].shape

    @property
    def spatial_shape(self):
        return self.cubes[0].spatial_shape

    def get_arrays(self, i_lon, i_lat):
        arrays = []
        for dcube in self.cubes:
            xarr, arr, *_ = dcube.get_spec_data(i_lon, i_lat)
            arrays.append(arr)
        return arrays

    def get_spec_data(self, i_lon, i_lat):
        all_spec_data = []
        any_nans = False
        for dcube in self.cubes:
            *spec_data, has_nans = dcube.get_spec_data(i_lon, i_lat)
            all_spec_data.append(spec_data)
            any_nans |= has_nans
        return all_spec_data, any_nans

    def get_max_snr(self, i_lon, i_lat):
        max_snr = 0.0
        for dcube in self.cubes:
            _, arr, noise, _, _ = dcube.get_spec_data(i_lon, i_lat)
            spec_snr = np.max(arr) / noise
            max_snr = spec_snr if spec_snr > max_snr else max_snr
        return max_snr


def check_ext(store_name, ext='hdf'):
    if store_name.endswith(f'.{ext}'):
        return store_name
    else:
        return f'{store_name}.{ext}'


class HdfStore:
    linked_table = Path('table.hdf')
    chunk_prefix = 'chunk'
    dpath = '/products'

    def __init__(self, store_name, nchunks=1):
        """
        Parameters
        ----------
        store_name : str
        nchunks : int
        """
        self.store_name = str(store_name)
        self.store_dir = Path(check_ext(self.store_name, ext='store'))
        self.store_dir.mkdir(parents=True, exist_ok=True)
        # FIXME Perform error handling for if HDF file is already open
        self.hdf = h5py.File(self.store_dir / self.linked_table, 'a')
        try:
            self.nchunks = self.hdf.attrs['nchunks']
        except KeyError:
            self.hdf.attrs['nchunks'] = nchunks
            self.nchunks = nchunks
        try:
            model_name = self.hdf.attrs['model_name']
            self.model = MODELS[model_name]
        except KeyError:
            self.model = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def chunk_paths(self):
        return [
                self.store_dir / Path(f'{self.chunk_prefix}{i}.hdf')
                for i in range(self.nchunks)
        ]

    @property
    def is_open(self):
        # If the HDF file is closed, it will raise an exception stating
        # "ValueError: Not a file (not a file)"
        try:
            self.hdf.mode
            return True
        except ValueError:
            return False

    def close(self):
        try:
            self.hdf.flush()
            self.hdf.close()
        except ValueError:
            print('Store HDF already closed.')

    def iter_pix_groups(self):
        assert self.is_open
        for lon_pix in self.hdf['/pix']:
            if lon_pix is None:
                raise ValueError(f'Broken external HDF link: /pix/{lon_pix}')
            for lat_pix in self.hdf[f'/pix/{lon_pix}']:
                if lat_pix is None:
                    raise ValueError(f'Broken external HDF link: /pix/{lon_pix}/{lat_pix}')
                group = self.hdf[f'/pix/{lon_pix}/{lat_pix}']
                if not isinstance(group, h5py.Group):
                    continue
                yield group

    def link_files(self):
        assert self.is_open
        for chunk_path in self.chunk_paths:
            with h5py.File(chunk_path, 'r') as chunk_hdf:
                for lon_pix in chunk_hdf['/pix']:
                    for lat_pix in chunk_hdf[f'/pix/{lon_pix}']:
                        group_name = f'/pix/{lon_pix}/{lat_pix}'
                        group = h5py.ExternalLink(chunk_path.name, group_name)
                        self.hdf[group_name] = group
                self.hdf.flush()

    def reset_pix_links(self):
        assert self.is_open
        if '/pix' in self.hdf:
            del self.hdf['/pix']

    def insert_header(self, stack):
        if self.is_open:
            sh_g = self.hdf.create_group('simple_header')
            for k, v in stack.simple_header.items():
                sh_g.attrs[k] = v
            fh_g = self.hdf.create_group('full_header')
            for k, v in stack.full_header.items():
                fh_g.attrs[k] = v
            self.hdf.attrs['naxis1'] = stack.shape[0]
            self.hdf.attrs['naxis2'] = stack.shape[1]
        else:
            warnings.warn(
                    'Could not insert header: the HDF5 file is closed.',
                    category=RuntimeWarning,
            )

    def read_header(self, full=True):
        assert self.is_open
        hdr_group_name = 'full_header' if full else 'simple_header'
        h_group = self.hdf[hdr_group_name]
        header = fits.Header()
        for k, v in h_group.attrs.items():
            header[k] = v
        return header

    def create_dataset(self, dset_name, data, group='', clobber=True):
        assert len(dset_name) > 0
        self.hdf.require_group(group)
        path = f'{group.rstrip("/")}/{dset_name}'
        if path in self.hdf and clobber:
            warnings.warn(f'Deleting dataset "{path}"', RuntimeWarning)
            del self.hdf[path]
        return self.hdf[group].create_dataset(dset_name, data=data)

    def insert_fitter_pars(self, fitter):
        assert self.is_open
        self.hdf.attrs['lnZ_threshold'] = fitter.lnZ_thresh
        self.hdf.attrs['n_max_components'] = fitter.ncomp_max
        self.hdf.attrs['multinest_kwargs'] = str(fitter.mn_kwargs)

    def insert_model_metadata(self, runner_cls):
        module = inspect.getmodule(runner_cls)
        assert self.is_open
        self.hdf.attrs['n_params'] = module.N
        self.hdf.attrs['model_name'] = module.NAME
        self.hdf.attrs['par_names'] = module.PAR_NAMES
        self.hdf.attrs['par_names_short'] = module.PAR_NAMES_SHORT
        self.hdf.attrs['tex_labels'] = module.TEX_LABELS
        self.hdf.attrs['tex_labels_with_units'] = module.TEX_LABELS_WITH_UNITS


class CubeFitter:
    mn_default_kwargs = {
            'nlive':   100,
            'tol':     1.0,
            'efr':     0.3,
            'updInt': 2000,
    }

    def __init__(self, stack, utrans, runner_cls, runner_kwargs=None,
            lnZ_thresh=11, ncomp_max=2, mn_kwargs=None, nlive_snr_fact=5):
        """
        Parameters
        ----------
        stack : CubeStack
        utrans : PriorTransformer
        runner_cls : Runner
        runner_kwargs : dict
        lnZ_thresh : number, default 11
            Log evidence threshold to use when choosing whether to increment
            the number of fit components.
        ncomp_max : number, default 2
            Maximum number of components to fit
        mn_kwargs : dict
            Keyword parameters to pass to MultiNest run call
        nlive_snr_fact : number, default 5
            Multiplicative factor to increase the number of live points used in
            fitting each pixel by:

                nlive + int(nlive_snr_fact * snr)

            where `nlive` is the value set in `mn_kwargs` arguement.
        """
        self.stack = stack
        self.utrans = utrans
        self.runner_cls = runner_cls
        self.runner_kwargs = {} if runner_kwargs is None else runner_kwargs
        self.lnZ_thresh = lnZ_thresh
        self.ncomp_max = ncomp_max
        self.mn_kwargs = self.mn_default_kwargs.copy()
        if mn_kwargs is not None:
            self.mn_kwargs.update(mn_kwargs)
        self.nlive_snr_fact = nlive_snr_fact

    def fit(self, *args):
        (all_lon, all_lat), chunk_path = args
        # NOTE for HDF5 files to be written correctly, they must be opened
        # *after* the `multiprocessing.Process` has been forked from the main
        # Python process, and it inherits the HDF5 libraries state.
        #   See "Python and HDF5" pg. 116
        # The NRAO Lustre system occasionally has issues opening files, leaving
        # the process to hang in the C layer of HDF5 without a traceback. I
        # suspect it is a timing issue when multiple user python processes
        # request new HDF files at exactly the same time. A small random time
        # is added in an attempt to avoid this.
        time.sleep(np.random.random())  # 0-1 sec
        hdf = h5py.File(chunk_path, 'a')
        for (i_lon, i_lat) in zip(all_lon, all_lat):
            spec_data, has_nans = self.stack.get_spec_data(i_lon, i_lat)
            if has_nans:
                # FIXME replace with logging framework
                print(f'-- ({i_lon}, {i_lat}) SKIP: has NaN values')
                continue
            group_name = f'/pix/{i_lon}/{i_lat}'
            group = hdf.require_group(group_name)
            # Increase number of live points based on the SNR factor
            max_snr = self.stack.get_max_snr(i_lon, i_lat)
            mn_kwargs = self.mn_kwargs.copy()
            mn_kwargs['nlive'] += int(self.nlive_snr_fact * max_snr)
            # Iteratively fit additional components until they no longer
            # produce a significant increase in the evidence.
            ncomp = 1
            nbest = 0
            while ncomp <= self.ncomp_max:
                print(f'-- ({i_lon}, {i_lat}) -> N = {ncomp}')
                sub_group = group.create_group(f'{ncomp}')
                dumper = Dumper(sub_group)
                runner = self.runner_cls.from_data(spec_data, self.utrans,
                        ncomp=ncomp, **self.runner_kwargs)
                if ncomp == 1:
                    old_lnZ = runner.null_lnZ
                    assert np.isfinite(old_lnZ)
                # FIXME needs tuned/specific kwargs for a given ncomp
                run_multinest(runner, dumper, **mn_kwargs)
                assert np.isfinite(runner.run_lnZ)
                if runner.run_lnZ - old_lnZ < self.lnZ_thresh:
                    break
                else:
                    old_lnZ = runner.run_lnZ
                    nbest = ncomp
                    ncomp += 1
            group.attrs['i_lon'] = i_lon
            group.attrs['i_lat'] = i_lat
            group.attrs['nbest'] = nbest
        hdf.flush()
        hdf.close()

    def fit_cube(self, store_name='run/test_cube', nproc=1, timeout=None):
        """
        Create an `HdfStore`, run NestFit on each pixel of the field, and
        write the results to the store file.

        Parameters
        ----------
        store_name : str
            The name or path of the `HdfStore` file to create and store the
            values in.
        nproc : int
            Number of Python processes to spawn using `multiprocessing`.
            The cubes are striped in longitude per-process, and thus the
            dimension of the longitude axis in pixels must be equal to
            or greater than the number of processes.
        timeout : bool, None
            Timeout to use in the Python `multiprocessing` call to join
            the processes. If `None`, no timeout is used.
        """
        n_chan, n_lat, n_lon = self.stack.shape
        if nproc > n_lon:
            raise ValueError(
                    f'The pixel width of the image in longitude ({n_lon}) ' +
                    f'must be greater than or equal to the number of processes ({nproc}).'
            )
        store = HdfStore(store_name, nchunks=nproc)
        store.insert_header(self.stack)
        store.insert_fitter_pars(self)
        store.insert_model_metadata(self.runner_cls)
        # create list of indices for each process
        indices = get_multiproc_indices(self.stack.spatial_shape, store.nchunks)
        if store.nchunks == 1:
            self.fit(indices[0], store.chunk_paths[0])
        else:
            # NOTE A simple `multiprocessing.Pool` cannot be used because the
            # Cython C-extensions cannot be pickled without implementing the
            # pickling protocol on all classes.
            # NOTE `mpi4py` may be more appropriate here, but it is more complex
            # FIXME no error handling if a process fails/raises an exception
            sequence = list(zip(indices, store.chunk_paths))
            procs = [
                    multiprocessing.Process(target=self.fit, args=args)
                    for args in sequence
            ]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(timeout)
        # link all of the HDF5 files together
        store.link_files()
        store.close()


def take_by_components(data, comps, axis=0, incl_zero=True):
    """
    Select the elements from the `data` array based on the component mask.
    Positions where the component mask is 0 or -1 (NaN sentinel value) are
    replaced by NaNs.

    Parameters
    ----------
    data : array-like, dimensions (..., b, l)
    comps : array-like, dimensions (b, l)
        Selected number of components. Note that values of -1 are interpreted
        as no-data.
    axis : int, default 0
        Axis in `data` to interpret the values in `comps` over.
    incl_zero : bool, default True
        Include null model when taking components
    """
    take = comps.copy()
    # convert number of components to array index
    #   PDF indices for 0 -> 1 comp; 1 -> 2 comp; etc.
    take -= 1
    take[take < 0] = 0
    new_axes = list(range(data.ndim - take.ndim))
    take = np.expand_dims(take, axis=new_axes)
    data = np.take_along_axis(data, take, axis=axis)
    data = np.squeeze(data, axis=axis)
    if incl_zero:
        # only exclude -1 for no-data
        mask = comps < 0
    else:
        # exclude both -1 for no-data and 0 for noise only
        mask = comps < 1
    data[...,mask] = np.nan
    return data


def get_multiproc_indices(shape, nproc):
    lon_ix, lat_ix = np.indices(shape)
    indices = [
            (lon_ix[i::nproc,...].flatten(), lat_ix[i::nproc,...].flatten())
            for i in range(nproc)
    ]
    return indices


def apply_circular_mask(kernel, radius=None):
    """
    Weight the kernel based on a sub-pixel exact solution for a uniform
    circular aperture.

    Parameters
    ----------
    kernel : array-like
        Kernel array to mask. Must be odd in both dimensions.
    radius : number
        Radius in pixels as defined from the mid-point of the center pixel.
        Does not need to be an integer. If `None` extend to limit of smallest
        axis.
    """
    nx, ny = kernel.shape
    if radius is None:
        radius = min(nx, ny) / 2
    # return unchanged mask if radius larger than kernel size
    corner_dist = np.sqrt((nx / 2)**2 + (ny / 2)**2)
    if radius > corner_dist:
        return kernel
    # kernel is only valid if odd shaped
    if nx % 2 == 0 or ny % 2 == 0:
        raise ValueError(f'Kernel dimensions must be odd: ({nx}, {ny})')
    try:
        from photutils.geometry.circular_overlap import circular_overlap_grid
    except ImportError:
        raise ImportError('Must have "photutils" installed.')
    xmin = -nx / 2
    xmax =  ny / 2
    ymin = -ny / 2
    ymax =  ny / 2
    use_exact = True
    subpixels = 5
    weights = circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, radius,
            use_exact, subpixels)
    return weights * kernel


def get_indep_info_kernel(sigma, nrad=1, sigma_taper=None):
    """
    Create a kernel for the amount of information independent of the center
    pixel position based on a Gaussian beam. Kernel is odd-shaped.

    Parameters
    ----------
    sigma : number
        Gaussian standard deviation in pixels. Currently implemented for a
        symmetric beam.
    nrad : int
        Radius of square from center pixel. Kernels dimensions are (2*nrad+1).
    sigma_taper : number
        Taper for downweighting large radii in the kernel. Standard deviation
        of the tapering kernel in pixels. If `None` apply no taper.
    """
    # TODO implement a full beam profile with major, minor, and pa
    assert isinstance(nrad, int) and nrad >= 0
    if nrad == 0:
        return np.array([[1.0]])
    ppbeam = 2 * np.pi * sigma**2
    # If the beam is smaller than a pixel, then unit information is still 1.
    ppbeam = max(1, ppbeam)
    i_n = 2 * nrad + 1
    Y, X = np.indices((i_n, i_n))
    X -= nrad
    Y -= nrad
    # Compute values from Gaussian function at pixel mid-point
    #kernel = 1 - np.exp(-0.5 * (X**2 + Y**2) / sigma**2)
    #kernel /= ppbeam
    #kernel[nrad, nrad] = 1
    #return kernel
    # Calculate bin edge boundaries
    X_lo = X - 0.5
    X_hi = X + 0.5
    Y_lo = Y - 0.5
    Y_hi = Y + 0.5
    # Integrate the beam profile over the pixel boundaries
    def phi(z):
        """Cumulative of the unit normal distribution"""
        return 0.5 * (1 + sp.special.erf(z / sigma / np.sqrt(2)))
    # Peak Gaussian amplitude factor, volume is the same over a 1x1 pixel
    peak_amp = 1 / (2 * np.pi * sigma**2)
    kernel = 1 - (phi(X_hi) - phi(X_lo)) * (phi(Y_hi) - phi(Y_lo)) / peak_amp
    kernel /= ppbeam
    if sigma_taper is not None:
        kernel *= np.exp(-0.5 * (X**2 + Y**2) / sigma_taper**2)
    kernel[nrad, nrad] = 1
    return kernel


def aggregate_run_attributes(store):
    """
    Aggregate the attribute values into a dense array from the individual
    per-pixel Nested Sampling runs. Products include:

        * 'nbest' (b, l)
        * 'evidence' (m, b, l)
        * 'evidence_err' (m, b, l)
        * 'AIC' (m, b, l)
        * 'AICc' (m, b, l)
        * 'BIC' (m, b, l)

    Parameters
    ----------
    store : HdfStore
    """
    print(':: Aggregating store attributes')
    hdf = store.hdf
    dpath = store.dpath
    n_lon = hdf.attrs['naxis1']
    n_lat = hdf.attrs['naxis2']
    ncomp_max = hdf.attrs['n_max_components']
    # dimensions (l, b, m) for evidence values
    #   (latitude, longitude, model)
    attrib_shape = (n_lon, n_lat, ncomp_max+1)
    lnz_data = nans(attrib_shape)
    lnzerr_data = nans(attrib_shape)
    bic_data = nans(attrib_shape)
    aic_data = nans(attrib_shape)
    aicc_data = nans(attrib_shape)
    # dimensions (l, b) for N-best
    nb_data = np.full((n_lon, n_lat), -1, dtype=np.int32)
    for group in store.iter_pix_groups():
        i_lon = group.attrs['i_lon']
        i_lat = group.attrs['i_lat']
        nbest = group.attrs['nbest']
        nb_data[i_lon,i_lat] = nbest
        for model in group:
            subg = group[model]
            ncomp = subg.attrs['ncomp']
            if ncomp == 1:
                lnz_data[i_lon,i_lat,0]  = subg.attrs['null_lnZ']
                bic_data[i_lon,i_lat,0]  = subg.attrs['null_BIC']
                aic_data[i_lon,i_lat,0]  = subg.attrs['null_AIC']
                aicc_data[i_lon,i_lat,0] = subg.attrs['null_AICc']
            lnz_data[i_lon,i_lat,ncomp] = subg.attrs['global_lnZ']
            lnzerr_data[i_lon,i_lat,ncomp] = subg.attrs['global_lnZ_err']
            bic_data[i_lon,i_lat,ncomp]  = subg.attrs['BIC']
            aic_data[i_lon,i_lat,ncomp]  = subg.attrs['AIC']
            aicc_data[i_lon,i_lat,ncomp] = subg.attrs['AICc']
    # transpose to dimensions (b, l)
    store.create_dataset('nbest', nb_data.transpose(), group=dpath)
    # transpose to dimensions (m, b, l)
    store.create_dataset('evidence', lnz_data.transpose(), group=dpath)
    store.create_dataset('evidence_err', lnzerr_data.transpose(), group=dpath)
    store.create_dataset('BIC', bic_data.transpose(), group=dpath)
    store.create_dataset('AIC', aic_data.transpose(), group=dpath)
    store.create_dataset('AICc', aicc_data.transpose(), group=dpath)


def convolve_evidence(store, kernel):
    """
    Convolve the evidence maps and re-select the preferred number of model
    components. Products include:

        * 'conv_evidence' (m, b, l)
        * 'conv_nbest' (b, l)

    Parameters
    ----------
    store : HdfStore
    kernel : number or `astropy.convolution.Kernel2D`
        Either a kernel instance or a number defining the standard deviation in
        map pixels of a Gaussian convolution kernel.
    """
    print(':: Convolving evidence maps')
    if isinstance(kernel, (int, float)):
        kernel = convolution.Gaussian2DKernel(kernel)
    hdf = store.hdf
    dpath = store.dpath
    ncomp_max = hdf.attrs['n_max_components']
    lnZ_thresh = hdf.attrs['lnZ_threshold']
    # dimensions (m, b, l)
    data = hdf[f'{dpath}/evidence'][...]
    # dimensions (b, l)
    nbest = hdf[f'{dpath}/nbest'][...]
    cdata = np.zeros_like(data)
    # Spatially convolve evidence values. The convolution operator is
    # distributive, so C(Z1-Z0) should equal C(Z1)-C(Z0).
    for i in range(data.shape[0]):
        cdata[i,:,:] = convolution.convolve(data[i,:,:], kernel, boundary='extend')
    # Re-compute N-best with convolved data
    conv_nbest = np.full(cdata[0].shape, 0, dtype=np.int32)
    for i in range(ncomp_max):
        # each step must pass the evidence threshold, eg both 0->1 and 1->2
        # where 1->2 should not be chosen if 0->1 was not.
        conv_nbest[
                (conv_nbest == i) &
                (cdata[i+1] - cdata[i] > lnZ_thresh)
        ] += 1
    # refill the "NaN" values
    conv_nbest[nbest == -1] = -1
    # Guard change in Nbest to no more than +1. In practice this should only apply
    # to a very small number of pixels but it will create errors because a jump
    # of +2 will not have had a model run for it.
    overshot = conv_nbest - nbest >= 2
    conv_nbest[overshot] = nbest[overshot] + 1
    # dimensions (b, l)
    store.create_dataset('conv_nbest', conv_nbest, group=dpath)
    # dimensions (m, b, l)
    store.create_dataset('conv_evidence', cdata, group=dpath)


def extended_masked_evidence(store, kernel, conv=True, lnz_thresh=3):
    """
    Mask the local or convolved evidence maps on a threshold for a second
    convolution to identify weak spatially extended features. Products include:

        * 'mext_evidence_diff' (b, l)

    Parameters
    ----------
    store : HdfStore
    kernel : number or `astropy.convolution.Kernel2D`
        Either a kernel instance or a number defining the standard deviation in
        map pixels of a Gaussian convolution kernel.
    conv : bool
        Use the convolved (`conv_evidence`) versus the local evidence (`evidence`)
    lnz_thresh : number
        Threshold to mask the initial evidence map on.
    """
    print(':: Convolving masked evidence')
    if isinstance(kernel, (int, float)):
        kernel = convolution.Gaussian2DKernel(kernel)
    hdf = store.hdf
    dpath = store.dpath
    # dimensions (m, b, l)
    data = hdf[f'{dpath}/evidence'][...]
    # dimensions (m, b, l)
    ev_name = 'conv_evidence' if conv else 'evidence'
    mdata = hdf[f'{dpath}/{ev_name}'][...]
    mdata = mdata[1] - mdata[0]
    mask = mdata > lnz_thresh
    # Spatially convolve the masked evidence values with the new kernel.
    cdata = nans(data.shape)
    for i in range(data.shape[0]):
        data[i,mask] = np.nan
        cdata[i,:,:] = convolution.convolve(data[i,:,:], kernel, boundary='extend')
    mext = cdata[1] - cdata[0]
    # refill the NaN values after the convolution interpolates over them
    mext[np.isnan(mdata) | mask] = np.nan
    # dimensions (b, l)
    store.create_dataset('mext_evidence', mext, group=dpath)


def aggregate_run_products(store):
    """
    Aggregate the results from the individual per-pixel Nested Sampling runs
    into dense arrays of the product values. Products include:

        * 'marg_quantiles' (M)
        * 'nbest_MAP' (m, p, b, l) -- cube of maximum a posteriori values
        * 'nbest_bestfit' (m, p, b, l) -- cube of maximum likelihood values
        * 'nbest_marginals' (m, p, M, b, l) -- marginal quantiles cube

    Parameters
    ----------
    store : HdfStore
    """
    print(':: Aggregating store products')
    hdf = store.hdf
    dpath = store.dpath
    n_lon = hdf.attrs['naxis1']
    n_lat = hdf.attrs['naxis2']
    # transpose from (b, l) -> (l, b) for consistency
    nbest_data = hdf[f'{dpath}/conv_nbest'][...].transpose()
    # get list of marginal quantile information out of store
    ncomp_max = hdf.attrs['n_max_components']
    n_params = hdf.attrs['n_params']
    test_group = hdf[f'pix/{n_lon//2}/{n_lat//2}/1']  # FIXME may not exist
    marg_quan = test_group.attrs['marg_quantiles']
    n_margs   = len(marg_quan)
    # dimensions (l, b, p, m) for MAP-parameter values
    mapdata = nans((n_lon, n_lat, n_params, ncomp_max))
    # dimensions (l, b, p, m) for bestfit parameter values
    bfdata = nans((n_lon, n_lat, n_params, ncomp_max))
    # dimensions (l, b, M, p, m) for posterior distribution marginals
    # NOTE in C order, the right-most index varies the fastest
    pardata = nans((n_lon, n_lat, n_margs, n_params, ncomp_max))
    # aggregate marginals into pardata
    for group in store.iter_pix_groups():
        i_lon = group.attrs['i_lon']
        i_lat = group.attrs['i_lat']
        nbest = nbest_data[i_lon,i_lat]
        if nbest == 0:
            continue
        nb_group = group[f'{nbest}']
        # convert MAP params from 1D array to 2D for:
        #   (p*m) -> (p, m)
        p_shape = (n_params, nbest)
        mapvs = nb_group['map_params'][...].reshape(p_shape)
        mapdata[i_lon,i_lat,:p_shape[0],:p_shape[1]] = mapvs
        # convert bestfit params from 1D array to 2D for:
        #   (p*m) -> (p, m)
        bfvs = nb_group['bestfit_params'][...].reshape(p_shape)
        bfdata[i_lon,i_lat,:p_shape[0],:p_shape[1]] = bfvs
        # convert the marginals output 2D array to 3D for:
        #   (M, p*m) -> (M, p, m)
        m_shape = (n_margs, n_params, nbest)
        margs = nb_group['marginals'][...].reshape(m_shape)
        pardata[i_lon,i_lat,:m_shape[0],:m_shape[1],:m_shape[2]] = margs
    # dimensions (M)
    store.create_dataset('marg_quantiles', marg_quan, group=dpath)
    # transpose to dimensions (m, p, b, l)
    store.create_dataset('nbest_MAP', mapdata.transpose(), group=dpath)
    # transpose to dimensions (m, p, b, l)
    store.create_dataset('nbest_bestfit', bfdata.transpose(), group=dpath)
    # transpose to dimensions (m, p, M, b, l)
    store.create_dataset('nbest_marginals', pardata.transpose(), group=dpath)


def aggregate_run_pdfs(store, par_bins=None):
    """
    Aggregate the results from the individual per-pixel set of posterior
    samples into one-dimensional marginalized PDFs of the parameter posteriors.
    Products include:

        * 'pdf_bins' (p, h)
        * 'post_pdfs' (r, m, p, h, b, l)

    Parameters
    ----------
    store : HdfStore
    par_bins : None or array (p, h+1)
        Histogram bin edges for each parameter. Note that the bin mid-points
        will be stored in the 'pdf_bins' array. If unset, then bins are created
        from the min and max values of the posteriors.
    """
    print(':: Aggregating store marginalized posterior PDFs')
    hdf = store.hdf
    dpath = store.dpath
    n_lon = hdf.attrs['naxis1']
    n_lat = hdf.attrs['naxis2']
    ncomp_max = hdf.attrs['n_max_components']
    n_params = hdf.attrs['n_params']
    # If no bins set, set bins from linear intervals of the posteriors
    if par_bins is None:
        n_bins = 200
        # Set linear bins from limits of the posterior marginal distributions.
        # Note that 0->min and 8->max in `Dumper.quantiles` and collapse all but
        # the second axis containing the model parameters.
        margdata = hdf[f'{dpath}/nbest_marginals'][...]
        vmins = np.nanmin(margdata[:,:,0,:,:], axis=(0,2,3))
        vmaxs = np.nanmax(margdata[:,:,8,:,:], axis=(0,2,3))
        par_bins = np.array([
                np.linspace(lo, hi, n_bins)
                for lo, hi in zip(vmins, vmaxs)
        ])
    else:
        n_bins = par_bins.shape[1]
    # dimensions (l, b, r, p, m, h) for histogram values
    #   (longitude, latitude, run, parameter, model, histogram-value)
    histdata = nans((n_lon, n_lat, ncomp_max, n_params, ncomp_max, n_bins-1))
    for group in store.iter_pix_groups():
        i_l = group.attrs['i_lon']
        i_b = group.attrs['i_lat']
        for i_r in range(ncomp_max):
            n_run = i_r + 1
            try:
                run_group = group[f'{n_run}']
            except KeyError:
                continue
            post = run_group['posteriors']
            for i_p, bins in enumerate(par_bins):
                for i_m in range(n_run):
                    ix = i_p * n_run + i_m
                    hist, _ = np.histogram(post[:,ix], bins=bins)
                    histdata[i_l,i_b,i_r,i_p,i_m,:] = hist
    # ensure the PDFs are normalized
    histdata /= np.nansum(histdata, axis=5, keepdims=True)
    # convert bin edges to bin mid-points
    # dimensions (m, h)
    bin_mids = (par_bins[:,:-1] + par_bins[:,1:]) / 2
    store.create_dataset('pdf_bins', bin_mids, group=dpath)
    # transpose from (l, b, r, p, m, h)
    #                 0  1  2  3  4  5
    #             to (r, m, p, h, b, l)
    #                 2  4  3  5  1  0
    histdata = histdata.transpose((2, 4, 3, 5, 1, 0)).astype('float32')
    store.create_dataset('post_pdfs', histdata, group=dpath)


def convolve_post_pdfs(store, kernel, evid_weight=True):
    """
    Spatially convolve the model posterior PDF. Products include:

        * 'conv_post_pdfs' (r, m, p, h, b, l)

    Parameters
    ----------
    store : HdfStore
    kernel : number or `astropy.convolution.Kernel2D`
        Either a kernel instance or a number defining the standard deviation in
        map pixels of a Gaussian convolution kernel.
    evid_weight : bool, default True
        Use the evidence over the null model to weight the pixel data.
    """
    print(':: Convolving posterior PDFs')
    if isinstance(kernel, (int, float)):
        kernel = convolution.Gaussian2DKernel(kernel)
    hdf = store.hdf
    dpath = store.dpath
    ncomp_max = hdf.attrs['n_max_components']
    # dimensions (r, m, p, h, b, l)
    data = hdf[f'{dpath}/post_pdfs'][...]
    cdata = np.zeros_like(data)
    # Fill zeros to avoid problems with zeros in log product
    data[data == 0] = 1e-32
    ldata = np.log(data)
    if evid_weight:
        # dimensions (m, b, l)
        evid = hdf[f'{dpath}/evidence'][...]
        # dimensions (b, l)
        nbest = hdf[f'{dpath}/conv_nbest'][...]
        # compute difference between preferred model number and zero.
        z_best = take_by_components(evid[1:,:,:], nbest)
        d_evid = z_best - evid[0,:,:]
        # transform to interval [0.0, 1.0]
        d_evid -= np.nanmin(d_evid)
        d_evid /= np.nanmax(d_evid)
        d_evid = d_evid.reshape((1, 1, 1, 1, *d_evid.shape))
        # weight the PDF distributions by the delta-evidence
        ldata *= d_evid
    # Spatially convolve the (l, b) map for every (model, parameter,
    # histogram) set.
    cart_prod = itertools.product(
            range(data.shape[0]),  # r
            range(data.shape[1]),  # m
            range(data.shape[2]),  # p
            range(data.shape[3]),  # h
    )
    for i_r, i_m, i_p, i_h in cart_prod:
        if i_m > i_r:
            continue
        cdata[i_r,i_m,i_p,i_h,:,:] = convolution.convolve_fft(
                ldata[i_r,i_m,i_p,i_h,:,:], kernel, normalize_kernel=False)
    # convert back to linear scaling
    cdata = np.exp(cdata)
    # ensure the PDFs are normalized
    cdata /= np.nansum(cdata, axis=3, keepdims=True)
    # re-mask the NaN positions
    cdata[np.isnan(data)] = np.nan
    cdata = cdata.astype('float32')
    store.create_dataset('conv_post_pdfs', cdata, group=dpath)


def quantize_conv_marginals(store):
    """
    Calculate weighted quantiles of convolved posterior marginal distributions.
    Products include:

        * 'conv_marginals' (r, m, p, M, b, l)

    Parameters
    ----------
    store : HdfStore
    """
    print(':: Calculating convolved PDF quantiles')
    hdf = store.hdf
    dpath = store.dpath
    # dimensions (p, h)
    bins = hdf[f'{dpath}/pdf_bins'][...]
    # dimensions (M)
    quan = hdf[f'{dpath}/marg_quantiles'][...]
    # dimensions (r, m, p, h, b, l)
    #   transposed to (r, m, p, b, l, h)
    data = hdf[f'{dpath}/conv_post_pdfs'][...]
    data = data.transpose((0, 1, 2, 4, 5, 3))
    data = np.cumsum(data, axis=5) / np.sum(data, axis=5, keepdims=True)
    # dimensions (r, m, p, b, l, M)
    margs_shape = list(data.shape)
    margs_shape[-1] = len(quan)  # h -> M
    margs = nans(margs_shape)
    # requires creating a new iterator each loop otherwise will run out
    def make_cart_prod():
        return itertools.product(
                range(data.shape[0]),  # r
                range(data.shape[1]),  # m
                range(data.shape[3]),  # b
                range(data.shape[4]),  # l
        )
    for i_p, x in enumerate(bins):
        for i_r, i_m, i_b, i_l in make_cart_prod():
            y = data[i_r,i_m,i_p,i_b,i_l]
            margs[i_r,i_m,i_p,i_b,i_l,:] = np.interp(quan, y, x)
    # transpose back to conventional shape (r, m, p, M, b, l)
    margs = margs.transpose((0, 1, 2, 5, 3, 4)).astype('float32')
    store.create_dataset('conv_marginals', margs, group=dpath)


def deblend_hf_intensity(store, stack, runner):
    """
    Calculate integrated and peak intensity maps from the maximum a posteriori
    parameter values. Also produce line profiles that have had the hyperfine
    splitting deblended.
    Products include:

        * 'peak_intensity' (t, m, b, l)
        * 'integrated_intensity' (t, m, b, l)
        * 'hf_deblended' (t, m, S, b, l)

    Parameters
    ----------
    store : HdfStore
    stack : CubeStack
    runner : Runner
    """
    assert runner.ncomp == 1
    print(':: Deblending HF structure in intensity map')
    hdf = store.hdf
    dpath = store.dpath
    # dimensions (p, h)
    bins = hdf[f'{dpath}/pdf_bins'][...]
    nbins = bins.shape[1]
    # dimensions (l, b, p, m)
    pmap = hdf[f'{dpath}/nbest_MAP'][...].transpose()
    # dimensions (l, b, m, t)
    #   for (lon, lat, model, transition)
    nspec = stack.n_cubes
    intint = nans((
            pmap.shape[0],
            pmap.shape[1],
            pmap.shape[3],
            nspec,
    ))
    pkint = nans(intint.shape)
    cart_prod = itertools.product(
            range(pmap.shape[0]),
            range(pmap.shape[1]),
            range(pmap.shape[3]),
    )
    cube_shape = stack.spatial_shape
    for i_l, i_b, i_m in cart_prod:
        params = pmap[i_l,i_b,:,i_m].copy()  # make contiguous
        if np.any(np.isnan(params)):
            continue
        runner.predict(params)
        for i_t, spec in enumerate(runner.get_spectra()):
            pkint[ i_l,i_b,i_m,i_t] = spec.max_spec
            intint[i_l,i_b,i_m,i_t] = spec.sum_spec
    # scale intensities by velocity channel width to put in K*km/s
    for i_t, cube in enumerate(stack.cubes):
        intint[:,:,:,i_t] *= cube.dv
    # Desired dimensions (l, b, m, t, S) for `hfdb` below. It is created by
    # broadcasting along the last axis (S) for the velocity bins.
    dv_bin = abs(bins[0,1] - bins[0,0])
    vaxis = bins[0].reshape(1, 1, 1, 1, -1)
    ix_vcen = store.model.IX_VCEN
    ix_sigm = store.model.IX_SIGM
    vcen = np.expand_dims(pmap[:,:,ix_vcen,:], (3, 4))
    sigm = np.expand_dims(pmap[:,:,ix_sigm,:], (3, 4))
    norm_fact = dv_bin / (sigm * np.sqrt(2 * np.pi))
    amp = intint[...,np.newaxis]
    hfdb = norm_fact * amp * np.exp(-0.5 * ((vaxis - vcen) / sigm)**2)
    # transpose to (t, m, b, l)
    store.create_dataset('peak_intensity', pkint.transpose(), group=dpath)
    store.create_dataset('integrated_intensity', intint.transpose(), group=dpath)
    # transpose (l, b, m, t, S) -> (t, m, S, b, l)
    hfdb = hfdb.transpose((3, 2, 4, 1, 0)).astype('float32')
    store.create_dataset('hf_deblended', hfdb, group=dpath)


def generate_predicted_profiles(store, stack, runner):
    """
    Calculate emergent intensity profiles for the maximum a posterior parameter
    values for each transition. Note that these products are N-velocity
    components times as large as the original data cubes, so can be large.
    Products include:

        * 'model_spec_trans<TRANS_ID>' (m, S, b, l)

    Parameters
    ----------
    store : HdfStore
    stack : CubeStack
    runner: Runner
    """
    assert runner.ncomp == 1
    print(':: Generating MAP model spectral profiles')
    hdf = store.hdf
    dpath = store.dpath
    nspec = stack.n_cubes
    cube_shape = stack.spatial_shape
    # dimensions (p, h)
    bins = hdf[f'{dpath}/pdf_bins'][...]
    nbins = bins.shape[1]
    # dimensions (l, b, p, m)
    pmap = hdf[f'{dpath}/nbest_MAP'][...].transpose()
    # The cubes do not necessarily have the same spectral axes, so separate
    # ndarrays must be used for each transition.
    # dimensions (l, b, m, S)
    #   for (lon, lat, model, channel)
    # Note that the spectral shape (S) of the *cube* is used, not the Prior,
    # as used making the deblended line profiles.
    model_cubes = [
            nans((
                pmap.shape[0],
                pmap.shape[1],
                pmap.shape[3],
                dcube.nchan,
            ))
            for dcube in stack
    ]
    cart_prod = itertools.product(
            range(pmap.shape[0]),
            range(pmap.shape[1]),
            range(pmap.shape[3]),
    )
    for i_l, i_b, i_m in cart_prod:
        params = pmap[i_l,i_b,:,i_m].copy()  # make array contiguous
        if np.any(np.isnan(params)):
            continue
        runner.predict(params)
        for mcube, spec in zip(model_cubes, runner.get_spectra()):
            mcube[i_l,i_b,i_m,:] = spec.get_spec()
    for mcube, dcube in zip(model_cubes, stack):
        # transpose (l, b, m, S) -> (m, S, b, l)
        mcube = mcube.transpose((2, 3, 1, 0)).astype('float32')
        group = f'{dpath}/model_spec'
        store.create_dataset(f'trans{dcube.trans_id}', mcube, group=group)


def create_fits_from_store(store, prefix='source'):
    """
    Create FITS files from the datasets in the HDF Store.

    Parameters
    ----------
    store : HdfStore
    """
    map_header = store.read_header(full=False)
    cube_header = store.read_header(full=True)
    hdf = store.hdf
    dpath = store.dpath
    # dimensions (p, h)
    bins = hdf[f'{dpath}/pdf_bins'][...]
    nbins = bins.shape[1]
    vaxis = bins[store.model.IX_VCEN]
    ### hyperfine deblended cube
    # dimensions (t, m, S, b, l)
    hfdb = hdf[f'{dpath}/hf_deblended'][...]
    hfdb = hfdb.transpose((1, 2, 0, 3, 4))
    n_t = hfdb.shape[0]
    for i_t in range(n_t):
        # sum axis `m` or model component number
        data = np.nansum(hfdb[i_t], axis=0)
        header = cube_header.copy()
        header.update({
            'BUNIT': 'K',
            'NAXIS3': vaxis.size,
            'CRPIX3': 1,  # fortran 1-based indexing
            'CDELT3': vaxis[1] - vaxis[0],
            'CUNIT3': 'km/s',
            'CTYPE3': 'VRAD',
            'CRVAL3': vaxis[0],
            'SPECSYS': 'LSRK',
        })
        hdu = fits.PrimaryHDU(data, header)
        hdu.writeto(f'{prefix}_hf_deblended_trans{i_t}.fits', overwrite=True)
    # TODO implement:
    #  - peak intensity
    #  - integrated intensity
    #  - PDF local (MAP, median, error)
    #  - PDF joint


def postprocess_run(store, stack, runner, par_bins=None, evid_kernel=None,
        post_kernel=None, evid_weight=True):
    """
    Run all post-processing steps on the store file. The individual pixel data
    is aggregated into dense array products and a spatial convolution is
    applied to the evidence and posteriors.

    Parameters
    ----------
    store : HdfStore
    stack : CubeStack
    runner : Runner
    par_bins : list-like
    evid_kernel : number or `astropy.convolution.Kernel2D`
        Either a kernel instance or a number defining the standard deviation in
        map pixels of a Gaussian convolution kernel. Kernel to be used for
        evidence convolution.
    post_kernel : number or `astropy.convolution.Kernel2D`
        Kernel used to be used for posterior distribution convolution. Note
        that a normalized kernel is unlikely to be desired, as this will result
        in a weighted geometric mean of the posteriors.
    evid_weight : bool, default True
        Use the evidence over the null model to weight the pixel data in the
        posterior distribution convolution.
    """
    aggregate_run_attributes(store)
    convolve_evidence(store, evid_kernel)
    aggregate_run_products(store)
    aggregate_run_pdfs(store, par_bins=par_bins)
    convolve_post_pdfs(store, post_kernel, evid_weight=evid_weight)
    quantize_conv_marginals(store)
    deblend_hf_intensity(store, stack, runner)
    generate_predicted_profiles(store, stack, runner)


##############################################################################
#                                 Tests
##############################################################################

def test_nested(ncomp=2, prefix='test', nlive=100, tol=1.0):
    synspec = get_test_spectra()
    spectra = np.array([syn.to_ammspec() for syn in synspec])
    amm1 = spectra[0]
    amm2 = spectra[1]
    utrans = get_irdc_priors(vsys=0)
    hdf_filen = 'test.hdf'
    if Path(hdf_filen).exists():
        os.remove(hdf_filen)
    with h5py.File(hdf_filen, 'a', driver='core') as hdf:
        group = hdf.require_group(f'{prefix}/{ncomp}')
        dumper = Dumper(group)
        runner = AmmoniaRunner(spectra, utrans, ncomp)
        run_multinest(runner, dumper, nlive=nlive, seed=5, tol=tol, efr=0.3,
                updInt=2000)
    return synspec, spectra, runner


def profile_nested(ncomp=2):
    """
    Profile the various components using the line profiler. Run using the
    IPython magic `lprun`:

        %lprun -f main.profile_nested main.profile_nested()
    """
    synspec = get_test_spectra()
    spectra = np.array([syn.to_ammspec() for syn in synspec])
    amm1 = spectra[0]
    amm2 = spectra[1]
    utrans = get_irdc_priors(vsys=0)
    with h5py.File('empty.hdf', 'a', driver='core') as hdf:
        group = hdf.require_group(f'test/{ncomp}')
        dumper = Dumper(group, no_dump=True)
        runner = AmmoniaRunner(spectra, utrans, ncomp)
        # optimal cache layout prediction without python overhead
        n_repeat = 1e4
        utheta = 0.5 * np.ones(6*ncomp)
        utrans.transform(utheta, ncomp)
        test_profile_predict(amm1, utheta, n_repeat=n_repeat)
        test_profile_predict(amm2, utheta, n_repeat=n_repeat)
        # transform and spectral prediction
        for _ in range(1000):
            utheta = np.random.uniform(0, 1, size=6*ncomp)
            utrans.transform(utheta, ncomp)
            amm_predict(amm1, utheta)
            amm_predict(amm2, utheta)
            amm1.loglikelihood
            amm2.loglikelihood
        # likelihood evaluation
        for _ in range(1000):
            utheta = np.random.uniform(0, 1, size=6*ncomp)
            runner.loglikelihood(utheta)
        # full MultiNest run
        for _ in range(20):
            run_multinest(runner, dumper, nlive=100, seed=-1, tol=1.0, efr=0.3,
                    updInt=2000)


def get_test_cubestack(full=False):
    # NOTE hack in indexing because last channel is all NaN's
    cube11 = spectral_cube.SpectralCube.read('data/test_cube_11.fits')[:-1]
    cube22 = spectral_cube.SpectralCube.read('data/test_cube_22.fits')[:-1]
    if not full:
        cube11 = cube11[:,155:195,155:195]
        cube22 = cube22[:,155:195,155:195]
    noise_map = NoiseMapUniform(rms=0.35)
    cubes = (
            DataCube(cube11, noise_map=noise_map, trans_id=1),
            DataCube(cube22, noise_map=noise_map, trans_id=2),
    )
    stack = CubeStack(cubes)
    return stack


def test_fit_cube(store_name='run/test_cube_multin'):
    store_filen = f'{store_name}.store'
    if Path(store_filen).exists():
        shutil.rmtree(store_filen)
    stack = get_test_cubestack(full=False)
    utrans = get_irdc_priors(vsys=63.7)  # correct for G23481 data
    fitter = CubeFitter(stack, utrans, AmmoniaRunner, ncomp_max=1)
    fitter.fit_cube(store_name=store_name, nproc=8)


def test_pyspeckit_profiling_compare(n=100):
    import pyspeckit
    # factors which provide constant overhead
    #params = np.array([-1.0, 10.0, 4.0, 14.5,  0.3,  0.0])
    #        ^~~~~~~~~ voff, trot, tex, ntot, sigm, orth
    utrans = get_irdc_priors()
    s11, s22 = get_test_spectra()
    xarr = s11.xarr.value.copy()
    data = s11.sampled_spec
    amms = AmmoniaSpectrum(xarr, data, 0.1, trans_id=1)
    # loop spectra to average function calls by themselves
    for _ in range(n):
        params = np.random.uniform(0, 1, size=6)
        utrans.transform(params, 1)
        amm_predict(amms, params)
        pyspeckit.spectrum.models.ammonia.ammonia(
                s11.xarr, xoff_v=params[0], trot=params[1], tex=params[2],
                ntot=params[3], width=params[4], fortho=params[5],
                line_names=['oneone'])


