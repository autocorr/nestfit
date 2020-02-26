#!/usr/bin/env python3
"""
Utilities for processing synthetic spectra.
"""

import numpy as np

import pyspeckit
from astropy import units as u
from astropy.io import fits

from .wrapped import AmmoniaSpectrum


FAKE_KWDS = {
        'WCSAXES': 3,
        'CRPIX1': np.nan,
        'CRPIX2': np.nan,
        'CRPIX3': np.nan,
        'CDELT1': 1e-4,
        'CDELT2': 1e-4,
        'CDELT3': np.nan,
        'CTYPE1': 'RA---CAR',
        'CTYPE2': 'DEC--CAR',
        'CTYPE3': 'FREQ',
        'CRVAL1': 0,
        'CRVAL2': 0,
        'CRVAL3': np.nan,
        'CUNIT1': 'deg',
        'CUNIT2': 'deg',
        'CUNIT3': 'Hz',
        'RESTFRQ': np.nan,
        'BUNIT':  'K',
        'LONPOLE': 0,
        'LATPOLE': 180,
        'EQUINOX': 2000.0,
        'SPECSYS': 'LSRK',
        'RADESYS': 'FK5',
        'SSYSOBS': 'TOPOCENT',
}


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


def make_fake_header(data, xarr):
    kwds = FAKE_KWDS.copy()
    kwds['CRPIX1'] = data.shape[0]
    kwds['CRPIX2'] = data.shape[1]
    kwds['CRPIX3'] = xarr.shape[0]
    kwds['CRVAL3'] = xarr.refX
    kwds['CDELT3'] = xarr[1] - xarr[0]
    kwds['RESTFRQ'] = xarr.refX
    return fits.Header(kwds)


def add_noise_to_cube(data, std):
    noise = np.random.normal(scale=std, size=data.size)
    return data + noise.reshape(data.shape)


class ParamSampler:
    def __init__(self, vsep=(0.16, 3), trot=(3, 30), tex=(2.8, 12),
            ntot=(13, 16), sigm=(0.15, 2)):
        """
        Parameters
        ----------
        vsep, trot, tex, ntot, sigm : tuple
            parameter ranges as (lo, hi) to uniformly sample from
        """
        self.vsep = vsep
        self.trot = trot
        self.tex  = tex
        self.ntot = ntot
        self.sigm = sigm

    def draw(self):
        vsep = np.random.uniform(*self.vsep)
        voff = np.array([0, vsep])
        return np.concatentate([
                voff,
                np.random.uniform(*self.trot, size=2),
                np.random.uniform(*self.tex,  size=2),
                np.random.uniform(*self.ntot, size=2),
                np.random.uniform(*self.sigm, size=2),
        ])


def make_indep_synth_cube():
    im_shape = (8, 4096)  # 8*4096 -> 32768; 8 rows for multi-processing purposes
    param_sampler = ParamSampler()
    # make synthetic cubes
    spectra = get_test_spectra()
    xarr11 = spectra[0].xarr
    xarr22 = spectra[1].xarr
    data11 = np.empty(np.product(im_shape) * xarr11.shape[0]).reshape(-1, xarr11.shape[0])
    data22 = np.empty(np.product(im_shape) * xarr22.shape[0]).reshape(-1, xarr22.shape[0])
    pcube  = np.empty(np.product(im_shape) * 10).reshape(-1, 10)  # p=5, n=2
    pkcube = np.empty(np.product(im_shape) * 2).reshape(-1, 2)
    # create synthetic cubes from parameters without noise
    for ii in range(np.product(im_shape)):
        params = param_sampler.draw()
        pcube[ii]  = params
        syn11 = SyntheticSpectrum(xarr11, params, noise=0, set_seed=False)
        syn22 = SyntheticSpectrum(xarr22, params, noise=0, set_seed=False)
        data11[ii] = syn11.sum_spec
        data22[ii] = syn22.sum_spec
        pkcube[ii] = (syn11.sum_spec.max(), syn.sum_spec.max())
    pcube = pcube.reshape(im_shape[0], im_shape[1], -1)
    fits.PrimaryHDU(pcube).writeto('syn_params.fits')
    pkcube = pkcube.reshape(im_shape, im_shape[1], -1)
    fits.PrimaryHDU(pkcube).writeto('syn_peak.fits')
    data11 = data11.reshape(im_shape[0], im_shape[1], -1)
    data22 = data22.reshape(im_shape[0], im_shape[1], -1)
    header11 = make_fake_header(data11, xarr11)
    header22 = make_fake_header(data11, xarr22)
    # add noise to cubes and then write to output FITS files
    for std in np.linspace(0.0, 0.5, 11):
        header11['RMS'] = std
        header22['RMS'] = std
        ndata11 = add_noise_to_cube(data11, std)
        ndata22 = add_noise_to_cube(data22, std)
        # create fits cube
        hdu11 = fits.PrimaryHDU(data=ndata11, header=header11)
        hdu22 = fits.PrimaryHDU(data=ndata22, header=header22)
        # write fits cube
        hdu11.writeto(f'syn_11_rms{noise:.3f}.fits')
        hdu22.writeto(f'syn_22_rms{noise:.3f}.fits')


