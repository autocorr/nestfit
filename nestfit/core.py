#!/usr/bin/env python3
"""
Spectral line decomposition using Nested Sampling.
"""

import shutil
import warnings
import multiprocessing
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import scipy as sp
import pandas as pd

import pyspeckit
import spectral_cube
from astropy.io import fits
from astropy import units as u

from .wrapped import (
        amm11_predict, amm22_predict,
        Prior, OrderedPrior, PriorTransformer,
        AmmoniaSpectrum, AmmoniaRunner,
        Dumper, run_multinest,
)


class SyntheticSpectrum:
    model_name = 'ammonia'

    def __init__(self, xarr, params, noise=0.03, vsys=0, set_seed=False):
        """
        Construct a mixture of ammonia model spectra given parameters:
            voff : centroid velocity offset from zero
            trot : rotation temperature
            tex  : excitation temperature
            ntot : para-ammonia column density
            sigm : velocity dispersion or gaussian sigma

        Parameters
        ----------
        xarr : pyspeckit.spectrum.units.SpectroscopicAxis
        params : np.ndarray
            1D array of parameters. Values are strided as [A1, A2, B1, B2, ...]
            for parameters A and B for components 1 and 2.
        noise : number, default 0.03
            Standard deviation of the baseline noise
        vsys : number, default 0
        set_seed : bool, default=False
            If `True` will use a default seed of 5 for the np.random module.
        """
        if set_seed:
            np.random.seed(5)
        else:
            np.random.seed()
        # ensure frequency ordering is ascending
        if xarr[1] - xarr[0] > 0:
            self.xarr = xarr.copy()
        else:
            self.xarr = xarr[::-1].copy()
        self.xarr.convert_to_unit('Hz')
        self.varr = self.xarr.as_unit('km/s')
        self.params = params
        self.noise = noise
        self.vsys = vsys
        self.size  = xarr.shape[0]
        self.ncomp = params.shape[0] // 5
        self.components = self.calc_profiles()
        self.sum_spec = self.components.sum(axis=0)
        self.noise_spec = self.calc_noise()
        self.sampled_spec = self.sum_spec + self.noise_spec

    def calc_profiles(self):
        n = self.ncomp
        models = np.array([
                pyspeckit.spectrum.models.ammonia.ammonia(
                    self.xarr,
                    xoff_v=self.params[    i]+self.vsys,
                    trot  =self.params[  n+i],
                    tex   =self.params[2*n+i],
                    ntot  =self.params[3*n+i],
                    width =self.params[4*n+i],
                    fortho=0.0,
                )
                for i in range(self.ncomp)
        ])
        return models

    def calc_noise(self):
        return np.random.normal(scale=self.noise, size=self.size)

    def resample_spectrum(self, noise=None):
        if noise is not None:
            self.noise = noise
        self.noise_spec = self.calc_noise()
        self.sampled_spec = self.sum_spec + self.noise_spec

    def to_ammspec(self):
        xarr = self.xarr.value.copy()
        data = self.sampled_spec
        return AmmoniaSpectrum(xarr, data, self.noise)


def get_test_spectra():
    freqs = pyspeckit.spectrum.models.ammonia_constants.freq_dict.copy()
    Axis = pyspeckit.spectrum.units.SpectroscopicAxis
    vchan = 0.158  # km/s
    vaxis = np.arange(-30, 30, vchan) * u.km / u.s
    xa11 = Axis(vaxis, velocity_convention='radio', refX=freqs['oneone']).as_unit('Hz')
    xa22 = Axis(vaxis, velocity_convention='radio', refX=freqs['twotwo']).as_unit('Hz')
    params = np.array([
        -1.0,  1.5,  # voff
        10.0, 15.0,  # trot
         4.0,  6.0,  # tex
        14.5, 15.0,  # ntot
         0.3,  0.6,  # sigm
    ])
    #params = np.array([
    #    -1.0,  1.0,  # voff
    #    12.0, 12.0,  # trot
    #     6.0,  6.0,  # tex
    #    14.5, 14.6,  # ntot
    #     0.3,  0.3,  # sigm
    #])
    spectra = [
            SyntheticSpectrum(xarr, params, noise=0.2, set_seed=True)
            for xarr in (xa11, xa22)
    ]
    return spectra


def get_irdc_priors(size=500, vsys=0.0):
    """
    Evaluate the inverse cumulative prior functions and construct a
    `PriorTransformer` instance for use with MultiNest. These distributions are
    constructed for the IRDCs observed in Svoboda et al. (in prep).

    Parameters
    ----------
    size : int
        Number of even, linearly spaced samples in the distribution
    vsys : float
        Systemic velocity to center prior distribution about
    """
    # prior distributions
    # NOTE gamma distributions evaluate to inf at 1, so only evaluate
    # functions up to 1-epsilon. For the beta distribution ppf, 1-epsilon
    # evaluates to 0.999045 .
    epsilon = 1e-13
    x = np.linspace(0, 1-epsilon, size)
    dist_voff = sp.stats.beta(5.0, 5.0)
    dist_trot = sp.stats.gamma(4.4, scale=0.070)
    dist_tex  = sp.stats.beta(1.0, 2.5)
    dist_ntot = sp.stats.beta(16.0, 14.0)
    dist_sigm = sp.stats.gamma(1.5, loc=0.03, scale=0.2)
    # interpolation values, transformed to the intervals:
    # voff [-4.00,  4.0] km/s  (centered on vsys)
    # trot [ 7.00, 30.0] K
    # tex  [ 2.74, 12.0] K
    # ntot [12.00, 17.0] log(cm^-2)
    # sigm [ 0.00,  2.0] km/s
    y_voff =  8.00 * dist_voff.ppf(x) -  4.00 + vsys
    y_trot = 23.00 * dist_trot.ppf(x) +  7.00
    y_tex  =  9.26 * dist_tex.ppf(x)  +  2.74
    y_ntot =  5.00 * dist_ntot.ppf(x) + 12.00
    y_sigm =  2.00 * dist_sigm.ppf(x)
    priors = [
            OrderedPrior(y_voff),
            Prior(y_trot),
            Prior(y_tex),
            Prior(y_ntot),
            Prior(y_sigm),
    ]
    return PriorTransformer(priors)


def test_nested(ncomp=2):
    synspec = get_test_spectra()
    spectra = [syn.to_ammspec() for syn in synspec]
    utrans = get_irdc_priors(vsys=0)
    dumper = Dumper(f'001/{ncomp}', store_name='test', no_dump=True)
    runner = AmmoniaRunner(spectra, utrans, ncomp)
    run_multinest(runner, dumper, nlive=60, seed=5, tol=1.0, efr=0.3,
            updInt=2000)
    return synspec, spectra, runner


class DataCube:
    def __init__(self, cube, noise=None):
        self.noise = noise
        self._header = cube.header.copy()
        self.data, self.xarr = self.data_from_cube(cube)
        self.shape = self.data.shape
        # NOTE data is transposed so (s, b, l) -> (l, b, s)
        self.spatial_shape = (self.shape[0], self.shape[1])

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
        hdict = {k: self._header[k] for k in keys}
        hdict['NAXIS'] = 2
        hdict['WCSAXES'] = 2
        coord_sys = ('ra', 'dec', 'lon', 'lat')
        # CTYPE's of form "RA---SIN"
        assert hdict['CTYPE1'].split('-')[0].lower() in coord_sys
        assert hdict['CTYPE2'].split('-')[0].lower() in coord_sys
        return hdict

    def data_from_cube(self, cube):
        # check ordering of 
        cube = cube.to('K').with_spectral_unit('Hz')
        axis = cube.spectral_axis.value.copy()
        nu_chan = axis[1] - axis[0]
        # frequency axis needs to be ascending order
        if nu_chan < 0:
            cube = cube[::-1]
            axis = cube.spectral_axis.value.copy()
        # data is transposed such that the frequency axis is contiguous (now
        # the last or right-most in of the indices)
        data = cube._data.transpose().copy()
        return data, axis

    def get_spectra(self, i_lon, i_lat):
        spec = self.data[i_lon,i_lat,:]  # axes reversed from typical cube
        has_nans = np.isnan(spec).any()
        if isinstance(self.noise, (float, int)):
            noise = self.noise
        else:
            noise = self.noise[i_lon,i_lat]
        amm_spec = AmmoniaSpectrum(self.xarr, spec, noise)
        return amm_spec, has_nans


class CubeStack:
    def __init__(self, cubes):
        assert cubes is not None
        self.cubes = cubes
        self.n_cubes = len(cubes)

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

    def get_spectra(self, i_lon, i_lat):
        spectra = []
        any_nans = False
        for dcube in self.cubes:
            spec, has_nans = dcube.get_spectra(i_lon, i_lat)
            spectra.append(spec)
            any_nans &= has_nans
        return spectra, any_nans


def check_ext(store_name, ext='hdf'):
    if store_name.endswith(f'.{ext}'):
        return store_name
    else:
        return f'{store_name}.{ext}'


class HdfStore:
    linked_table = Path('table.hdf')
    chunk_filen = 'chunk'

    def __init__(self, name, nchunks=1):
        self.store_name = name
        self.store_dir = Path(check_ext(self.store_name, ext='store'))
        if self.store_dir.exists():
            self.hdf = h5py.File(self.store_dir / self.linked_table, 'a')
            self.nchunks = self.hdf.attrs['nchunks']
        else:
            self.store_dir.mkdir()
            self.hdf = h5py.File(self.store_dir / self.linked_table, 'a')
            self.hdf.attrs['nchunks'] = nchunks
            self.nchunks = nchunks
            for chunk_name in self.chunk_names:
                with h5py.File(chunk_name, 'w') as hdf:
                    pass

    def __exit__(self, v_type, value, traceback):
        self.hdf.close()

    @property
    def chunk_names(self):
        paths = [
                self.store_dir / Path(f'{self.chunk_filen}{i}.hdf')
                for i in range(self.nchunks)
        ]
        return paths

    @property
    def is_open(self):
        try:
            self.hdf.mode
            return True
        except ValueError:
            # If the HDF file is closed, it will raise an exception stating
            # "ValueError: Not a file (not a file)"
            return False

    def iter_groups(self):
        for group_name in self.hdf:
            if group_name is None:
                raise ValueError('Group name is `None`: likely broken soft link in HDF')
            group = hdf[group_name]
            if not isinstance(group, h5py.Group):
                continue
            yield group

    def link_files(self):
        for chunk_path in self.chunk_names:
            with h5py.File(chunk_path, 'r') as chunk_hdf:
                for group_name in chunk_hdf:
                    self.hdf[group_name] = h5py.ExternalLink(chunk_path, group_name)

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

    def insert_fitter_pars(self, mappable):
        assert self.is_open
        self.hdf.attrs['lnZ_threshold'] = mappable.lnZ_thresh
        self.hdf.attrs['n_max_components'] = mappable.ncomp_max
        self.hdf.attrs['multinest_kwargs'] = str(mappable.mn_kwargs)


class MappableFitter:
    mn_default_kwargs = {
            'nlive':    60,
            'tol':     1.0,
            'efr':     0.3,
            'updInt': 2000,
    }

    def __init__(self, stack, utrans, lnZ_thresh=11, ncomp_max=2,
            mn_kwargs=None):
        self.stack = stack
        self.utrans = utrans
        self.lnZ_thresh = lnZ_thresh
        self.ncomp_max = ncomp_max
        self.mn_kwargs = mn_kwargs if mn_kwargs is not None else self.mn_default_kwargs

    def fit(self, *args):
        (all_lon, all_lat), store_path = args
        for (i_lon, i_lat) in zip(all_lon, all_lat):
            spectra, has_nans = self.stack.get_spectra(i_lon, i_lat)
            # FIXME replace with logging framework
            if has_nans:
                print(f':: ({i_lon}, {i_lat}) SKIP: has NaN values')
                continue
            group_name = f'pix_{i_lon}_{i_lat}'
            old_lnZ = -1e100
            new_lnZ = AmmoniaRunner(spectra, self.utrans, 1).null_lnZ
            assert np.isfinite(new_lnZ)
            ncomp = 0
            # Iteratively fit additional components until they no longer
            # produce a significant increase in the evidence.
            while new_lnZ - old_lnZ > self.lnZ_thresh:
                if ncomp == self.ncomp_max:
                    break
                ncomp += 1
                print(f':: ({i_lon}, {i_lat}) -> N = {ncomp}')
                sub_group_name = f'{group_name}/{ncomp}'
                dumper = Dumper(sub_group_name, store_name=str(store_path))
                runner = AmmoniaRunner(spectra, self.utrans, ncomp)
                # FIXME needs right kwargs for a given ncomp
                run_multinest(runner, dumper, **self.mn_kwargs)
                assert np.isfinite(runner.run_lnZ)
                old_lnZ, new_lnZ = new_lnZ, runner.run_lnZ
            with h5py.File(store_path, 'a') as hdf:
                group = hdf[group_name]
                group.attrs['i_lon'] = i_lon
                group.attrs['i_lat'] = i_lat
                group.attrs['nbest'] = ncomp - 1


def get_multiproc_indices(shape, nproc):
    lat_ix, lon_ix = np.indices(shape)
    indices = [
            (lon_ix[...,i::nproc].flatten(), lat_ix[...,i::nproc].flatten())
            for i in range(nproc)
    ]
    return indices


def fit_cube(stack, utrans, store_name='run/test_cube', nproc=1):
    n_chan, n_lat, n_lon = stack.shape
    mappable = MappableFitter(stack, utrans)
    store = HdfStore(store_name, nchunks=nproc)
    store.insert_header(stack)
    store.insert_fitter_pars(mappable)
    # create list of indices for each process
    indices = get_multiproc_indices(stack.spatial_shape, nproc)
    if nproc == 1:
        mappable.fit(indices[0], store.chunk_names[0])
    else:
        # spawn processes and process cube
        # NOTE A simple `multiprocessing.Pool` cannot be used because the
        # Cython C-extensions cannot be pickled without effort to implement
        # the pickling protocol on all classes.
        # FIXME no error handling if a process fails/raises an exception
        sequence = list(zip(indices, store.chunk_names))
        procs = [
                multiprocessing.Process(target=mappable.fit, args=args)
                for args in sequence
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
    # link all of the HDF5 files together
    store.link_files()


def test_fit_cube(store_name='run/test_cube_multin'):
    store_filen = f'{store_name}.store'
    if Path(store_filen).exists():
        shutil.rmtree(store_filen)
    cube11 = spectral_cube.SpectralCube.read('data/test_cube_11.fits')[:-1,155:195,155:195]
    cube22 = spectral_cube.SpectralCube.read('data/test_cube_22.fits')[:-1,155:195,155:195]
    cubes = (DataCube(cube11, noise=0.35), DataCube(cube22, noise=0.35))
    stack = CubeStack(cubes)
    utrans = get_irdc_priors(vsys=63.7)
    fit_cube(stack, utrans, store_name=store_name, nproc=8)


def marginals_to_ndcube(store):
    hdf = store.hdf
    n_lat = hdf.attrs['naxis1']
    n_lon = hdf.attrs['naxis2']
    # get list of parameters out of cube
    # get list of marginal percentiles
    ncomp_max = hdf.attrs['n_max_components']
    n_params  = hdf['pix_0_0/1'].attrs['n_params']
    par_names = hdf['pix_0_0/1'].attrs['par_names']
    marg_cols = hdf['pix_0_0/1'].attrs['marg_cols']
    marg_quan = hdf['pix_0_0/1'].attrs['marg_quantiles']
    n_margs   = len(marg_cols)
    # dimensions (l, b, M, p, m) for
    #   (latitude, longitude, marginal, parameter, model)
    # in C order, the right-most index varies the fastest
    pardata = np.empty((n_lon, n_lat, n_margs, n_params, ncomp_max))
    pardata[...] = np.nan
    # aggregate marginals into pardata
    for group in store.iter_groups():
        # TODO do similar agg/reshape for best-fit params
        i_lon = group.attrs['i_lon']
        i_lat = group.attrs['i_lat']
        nbest = group.attrs['nbest']
        nb_group = group[f'{nbest}']
        shape = (n_margs, n_params, nbest)
        # convert the marginals output 2D array for:
        #   (M, p*m) -> (M, p, m)
        margs = nb_group['marginals'][...].reshape(shape)
        pardata[i_lon,i_lat,:shape[0],:shape[1],:shape[2]] = margs
    # transpose to dimensions (m, p, M, b, l) and then keep multi-dimensional
    # parameter cube in the HDF5 file at the native dimensions
    pardata = pardata.transpose()
    hdf.create_dataset('nbest_marginal_cube', data=pardata)
    # FIXME
    # create header, reshape, then store as FITS
    #fits_header = fits.Header()
    #header_group = hdf['simple_header']
    #n_newax = n_margs * n_params * ncomp_max
    #for k, v in header_group:
    #    fits_header[k] = header_group[k]
    #fits_header.update({
    #    'NAXIS':  3,
    #    'NAXIS3': n_newax,
    #    'CTYPE3': 'PARAM',
    #    'CRVAL3': 1.,
    #    'CDELT3': 1.,
    #    'CRPIX3': 1.,
    #    'CUNIT3': '',
    #})
    #fits_data = pardata.reshape((n_newax, n_lon, n_lat))
    ## copy header and edit for 3-axis parameter cube
    #hdu = fits.PrimaryHDU(data=fits_data, header=fits_header)
    #hdu.writeto(f'{hdf.store_name}_marginals.fits')


def test_pyspeckit_profiling_compare(n=100):
    # factors which provide constant overhead
    s11, s22 = get_test_spectra()
    xarr = s11.xarr.value.copy()
    data = s11.sampled_spec
    params = np.array([-1.0, 10.0, 4.0, 14.5,  0.3])
    #        ^~~~~~~~~ voff, trot, tex, ntot, sigm
    amms = AmmoniaSpectrum(xarr, data, 0.1)
    # loop spectra to average function calls by themselves
    for _ in range(n):
        pyspeckit.spectrum.models.ammonia.ammonia(
                s11.xarr, xoff_v=-1.0, trot=10.0, tex=4.0, ntot=14.5,
                width=0.3, fortho=0, line_names=['oneone'])
        amm11_predict(amms, params)


