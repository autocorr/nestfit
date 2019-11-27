#!/usr/bin/env python3
"""
Gaussian mixture fitting with Nested Sampling.
"""

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


class CubeStack:
    freqs = pyspeckit.spectrum.models.ammonia_constants.freq_dict.copy()

    def __init__(self, cube11, cube22, noise=0.320):
        # FIXME handle noise per pixel from an explicit error map
        self._header = cube11.header.copy()
        self.cube11 = cube11.to('K')._data.copy()
        self.cube22 = cube22.to('K')._data.copy()
        self.noise  = noise
        self.shape = cube11.shape
        self.spatial_shape = (self.shape[1], self.shape[2])
        self.xarr11 = pyspeckit.spectrum.units.SpectroscopicAxis(
                cube11.spectral_axis,
                velocity_convention='radio',
                refX=self.freqs['oneone'],
        ).as_unit('Hz').value.copy()
        self.xarr22 = pyspeckit.spectrum.units.SpectroscopicAxis(
                cube22.spectral_axis,
                velocity_convention='radio',
                refX=self.freqs['twotwo'],
        ).as_unit('Hz').value.copy()

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
        hdict['NAXES'] = 2
        hdict['WCSAXES'] = 2
        coord_sys = ('ra', 'dec', 'lon', 'lat')
        assert hdict['CTYPE1'].lower() in coord_sys
        assert hdict['CTYPE2'].lower() in coord_sys
        return hdict

    @property
    def full_header(self):
        return self._header

    def get_spectra(self, i_lon, i_lat):
        data11 = self.cube11[:,i_lat,i_lon]
        data22 = self.cube22[:,i_lat,i_lon]
        spectra = [
                AmmoniaSpectrum(self.xarr11, data11, self.noise),
                AmmoniaSpectrum(self.xarr22, data22, self.noise),
        ]
        return spectra


def check_ext(store_name, ext='hdf'):
    if store_name.endswith(f'.{ext}'):
        return store_name
    else:
        return f'{store_name}.{ext}'


class HdfStore:
    linked_table = Path('table.hdf5')
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

    def __enter__(self):
        return self

    def __exit__(self, v_type, value, traceback):
        self.hdf.close()

    @property
    def chunk_names(self):
        paths = [
                self.store_dir / Path(f'{self.chunk_filen}{i}.hdf')
                for i in range(self.nchunks)
        ]
        return paths

    def link_files(self):
        for chunk_path in self.chunk_names:
            with h5py.File(chunk_path, 'r') as chunk_hdf:
                for group_name in chunk_hdf:
                    self.hdf[group_name] = h5py.ExternalLink(chunk_path, group_name)

    @property
    def is_open(self):
        try:
            self.hdf.mode
            return True
        except ValueError:
            # If the HDF file is closed, it will raise an exception stating
            # "ValueError: Not a file (not a file)"
            return False

    def insert_header(self, stack):
        if self.is_open:
            sh_g = self.hdf.create_group('simple_header')
            for k, v in stack.simple_header.items():
                sh_g.attrs[k] = v
            fh_g = self.hdf.create_group('full_header')
            for k, v in stack.full_header.items():
                fh_g.attrs[k] = v
        else:
            warnings.warn(
                    'Could not insert header: the HDF5 file is closed.',
                    category=RuntimeWarning,
            )


class MappableFitter:
    def __init__(self, stack, utrans, lnZ_thresh=11, ncomp_max=3,
            mn_kwargs=None):
        self.stack = stack
        self.utrans = utrans
        self.lnZ_thresh = lnZ_thresh
        self.ncomp_max = ncomp_max
        self.mn_kwargs = mn_kwargs if mn_kwargs is not None else {}

    def fit(self, *args):
        (all_lon, all_lat), store_path = args
        for (i_lon, i_lat) in zip(all_lon, all_lat):
            spectra = self.stack.get_spectra(i_lon, i_lat)
            group_name = f'pix_{i_lon}_{i_lat}'
            old_lnZ = -1e100
            new_lnZ = np.sum([s.null_lnZ for s in spectra])
            ncomp = 0
            # Iteratively fit additional components until they no longer
            # produce a significant increase in the evidence.
            while new_lnZ - old_lnZ > self.lnZ_thresh:
                ncomp += 1
                if ncomp == self.ncomp_max:
                    break
                print(f':: ({i_lon}, {i_lat}) -> N = {ncomp}')
                sub_group_name = group_name + f'/{ncomp}'
                dumper = HdfDumper(sub_group_name, store_name=str(store_path))
                runner = AmmoniaRunner(spectra, self.utrans, ncomp)
                # FIXME needs right kwargs per ncomp
                run_multinest(runner, dumper, **self.mn_kwargs)
                old_lnZ, new_lnZ = new_lnZ, dumper.lnZ
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
    store = HdfStore(store_name, nchunks=nproc)
    store.insert_header(stack)
    mappable = MappableFitter(stack, utrans)
    # create list of indices for each process
    indices = get_multiproc_indices(stack.spatial_shape, nproc)
    sequence = list(zip(indices, store.chunk_names))
    # spawn processes and process cube
    # NOTE A simple `multiprocessing.Pool` cannot be used because the
    # Cython C-extensions cannot be pickled without effort to implement
    # the pickling protocol on all classes.
    # FIXME no error handling if a process fails/raises an exception
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


def test_fit_cube():
    cube11 = spectral_cube.SpectralCube.read('data/test_cube_11.fits')[:-1]
    cube22 = spectral_cube.SpectralCube.read('data/test_cube_22.fits')[:-1]
    stack = CubeStack(cube11, cube22, noise=0.320)
    store_name = 'run/test_cube_multin'
    utrans = get_irdc_priors()
    fit_cube(stack, utrans, store_name=store_name, nproc=8)


def parse_results_to_cube(store_name, header):
    header = header.copy()
    n_lat = header['NAXIS1']
    n_lon = header['NAXIS2']
    pardata = np.empty((n_lat, n_lon))
    pardata[:,:] = np.nan
    if not store_name.endswith('.hdf5'):
        filen = store_name + '.hdf5'
    with h5py.File(filen, 'r') as hdf:
        for i_lat in range(n_lat):
            for i_lon in range(n_lon):
                group_name = f'pix_{i_lon}_{i_lat}'
                nbest = hdf[group_name].attrs['nbest']
                if nbest == 0:
                    continue
                #group = hdf[f'{group_name}/{nbest}']
                #post = group['posteriors']
                #ncolsum = np.nan
                #if nbest == 1:
                #    v1 = post[:,3]
                #    ncolsum = np.median(v1)
                #elif nbest == 2:
                #    v1 = post[:,6]
                #    v2 = post[:,7]
                #    vals = np.log10(10**v1 + 10**v2)
                #    ncolsum = np.median(vals)
                group = hdf[f'{group_name}/2']
                post = group['posteriors']
                ncolsum = np.nan
                v1 = post[:,6]
                v2 = post[:,7]
                vals = np.log10(10**v1 + 10**v2)
                ncolsum = np.median(vals)
                pardata[i_lat,i_lon] = ncolsum
    hdu = fits.PrimaryHDU(pardata)
    hdu.writeto(f'{store_name}_ncolsum.fits')


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


