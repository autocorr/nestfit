#!/usr/bin/env python3
"""
Gaussian mixture fitting with Nested Sampling.
"""

import h5py
import numpy as np
import pandas as pd

import corner
import pyspeckit
import pymultinest
from astropy import units as u

from nestfit.wrapped import (amm11_predict, amm22_predict, PriorTransformer,
        AmmoniaRunner)


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
        self.xarr = xarr.copy()
        self.varr = self.xarr.as_unit('km/s')
        self.xarr.convert_to_unit('Hz')
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


def test_spectra():
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


class AmmoniaSpectrum:
    def __init__(self, psk_spec, noise, name=None):
        assert noise > 0
        self.psk = psk_spec
        self.noise = noise
        self.name = name
        self.data = psk_spec.data.data
        self.size = psk_spec.shape[0]
        self.xarr = psk_spec.xarr.as_unit('Hz')
        self.prefactor = -self.size / 2 * np.log(2 * np.pi * noise**2)
        self.null_lnZ = self.loglikelihood(np.sum(self.data**2))

    def loglikelihood(self, sumsqdev):
        return self.prefactor - sumsqdev / (2 * self.noise**2)


def run_nested(runner, dumper, mn_kwargs=None):
    if mn_kwargs is None:
        mn_kwargs = {
                #'n_clustering_params': runner.ncomp,
                'outputfiles_basename': 'run/chain1-',
                'importance_nested_sampling': False,
                'multimodal': True,
                #'const_efficiency_mode': True,
                'n_live_points': 60,
                'evidence_tolerance': 0.5,
                'sampling_efficiency': 0.3,
                'n_iter_before_update': 2000,
                'verbose': True,
                'resume': False,
                'write_output': False,
                'dump_callback': dumper.dump,
        }
    pymultinest.run(
            runner.loglikelihood, None, runner.n_params, **mn_kwargs
    )


def test_nested(ncomp=2):
    synspec = test_spectra()
    spectra = [
        AmmoniaSpectrum(
            pyspeckit.Spectrum(xarr=syn.xarr, data=syn.sampled_spec, header={}),
            syn.noise, name)
        for syn, name in zip(synspec, ('oneone', 'twotwo'))
    ]
    utrans = PriorTransformer()
    dumper = HdfDumper('test_001')
    runner = AmmoniaRunner(spectra, utrans, ncomp)
    run_nested(runner, dumper)
    dumper.write_hdf()
    return synspec, spectra, runner


class HdfDumper:
    quantiles = np.array([
        0.00, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00,
        1.58655254e-1, 0.84134475,  # 1-sigma credible interval
        2.27501319e-2, 0.97724987,  # 2-sigma credible interval
        1.34989803e-3, 0.99865010,  # 3-sigma credible interval
    ])
    marginal_cols = [
        'min', 'p01', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99', 'max',
        '1s_lo', '1s_hi', '2s_lo', '2s_hi', '3s_lo', '3s_hi',
    ]

    def __init__(self, group_name, store_name='results'):
        self.group_name = group_name
        if not store_name.endswith('hdf5'):
            store_name += '.hdf5'
        self.store_name = store_name
        # These attributes are written to on each to call to `dump`
        self.n_calls = 0
        self.n_samples = None
        self.n_live = None
        self.n_params = None
        self.max_loglike = None
        self.lnZ = None
        self.lnZ_err = None
        self.posteriors = None

    def dump(self, n_samples, n_live, n_params, phys_live, posteriors,
            param_constr, max_loglike, lnZ, ins_lnZ, lnZ_err, null_context):
        self.n_calls += 1
        # The last two iterations will have the same number of samples, so
        # only copy over the parameters on the last iteration.
        if self.n_samples == n_samples:
            self.n_samples   = n_samples
            self.n_live      = n_live
            self.n_params    = n_params
            self.max_loglike = max_loglike
            self.lnZ         = lnZ
            self.lnZ_err     = lnZ_err
            # NOTE The array must be copied because MultiNest will free
            # the memory accessed by this view.
            self.posteriors  = posteriors.copy()
        else:
            self.n_samples = n_samples

    def calc_marginals(self):
        # The last two columns of the posterior array are -2*lnL and X*L/Z
        return np.quantile(self.posteriors[:,:-2], self.quantiles, axis=0)

    def write_hdf(self):
        with h5py.File(self.store_name, 'a') as hdf:
            group = hdf.create_group(self.group_name)
            # general run attributes:
            # TODO model_name, ncomp, par_labels, std_noise, null_lnZ
            # TODO calculate the AIC/AICc/BIC
            group.attrs['n_samples']      = self.n_samples
            group.attrs['n_live']         = self.n_live
            group.attrs['n_params']       = self.n_params
            group.attrs['global_lnZ']     = self.lnZ
            group.attrs['global_lnZ_err'] = self.lnZ_err
            group.attrs['max_loglike']    = self.max_loglike
            # posterior samples and statistics
            # TODO bestfit, MAP
            group.attrs['marginal_cols']  = self.marginal_cols
            group.create_dataset('posteriors', data=self.posteriors)
            group.create_dataset('marginals', data=self.calc_marginals())


if __name__ == '__main__':
    pass


