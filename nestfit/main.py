#!/usr/bin/env python3
"""
Gaussian mixture fitting with Nested Sampling.
"""

from pathlib import Path

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
        Construct a mixture of Gaussians expressed as:
            f(x) = A * exp(-(x - c)^2 / (2 * s^2))
        for "A" amplitude, "c" centroid, and "s" standard deviation.

        Parameters
        ----------
        xarr : pyspeckit.spectrum.units.SpectroscopicAxis
        params : np.ndarray
        noise : number, default 0.03
            Noise standard deviation
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


def run_nested(runner, basename='run/test_run_', call_prior=False, mn_kwargs=None):
    if mn_kwargs is None:
        mn_kwargs = {
                'outputfiles_basename': basename,
                'importance_nested_sampling': False,
                'multimodal': True,
                'evidence_tolerance': 0.5,
                'n_live_points': 100,
                'n_clustering_params': runner.ncomp,
                #'const_efficiency_mode': True,
                'sampling_efficiency': 0.3,
                'n_iter_before_update': 2000,
                'resume': False,
                'verbose': True,
        }
    if call_prior:
        pymultinest.run(
                runner.loglikelihood, runner.prior_transform,
                runner.n_params, **mn_kwargs
        )
    else:
        pymultinest.run(
                runner.loglikelihood, None, runner.n_params, **mn_kwargs
        )
    analyzer = pymultinest.Analyzer(
            outputfiles_basename=basename,
            n_params=runner.n_params,
    )
    lnZ = analyzer.get_stats()['global evidence']
    print(':: Evidence Z:', lnZ/np.log(10))
    return analyzer


def test_nested(ncomp=2):
    synspec = test_spectra()
    spectra = [
        AmmoniaSpectrum(
            pyspeckit.Spectrum(xarr=syn.xarr, data=syn.sampled_spec, header={}),
            syn.noise, name)
        for syn, name in zip(synspec, ('oneone', 'twotwo'))
    ]
    utrans = PriorTransformer()
    runner = AmmoniaRunner(
            spectra,
            utrans,
            ncomp,
    )
    analyzer = run_nested(runner)
    return synspec, spectra, runner, analyzer


def marginals_to_pandas(a_stats):
    margs = a_stats['marginals']
    df = pd.DataFrame(margs)
    new_cols = {
            'median': 'q50',
            'q01%':   'q01',
            'q10%':   'q10',
            'q25%':   'q25',
            'q75%':   'q75',
            'q90%':   'q90',
            'q99%':   'q99',
            '1sigma': 'ci_1sigma',
            '2sigma': 'ci_2sigma',
            '3sigma': 'ci_3sigma',
            '5sigma': 'ci_5sigma',
    }
    df = df.rename(columns=new_cols)
    df = df[[
        'q01', 'q10', 'q25', 'q50', 'q75', 'q90', 'q99',
        'sigma', 'ci_1sigma', 'ci_2sigma', 'ci_3sigma', 'ci_5sigma',
    ]]
    for col in ('ci_1sigma', 'ci_2sigma', 'ci_3sigma', 'ci_5sigma'):
        df[col+'_lo'] = df[col].apply(lambda x: x[0])
        df[col+'_hi'] = df[col].apply(lambda x: x[1])
        del df[col]
    return df


def save_run(runner, analyzer, group_name, store_name='nestfit'):
    # FIXME compute AIC/BIC here from model and max_loglike
    if not store_name.endswith('.hdf5'):
        store_name += '.hdf5'
    a_stats = analyzer.get_stats()
    bestfit = analyzer.get_best_fit()
    marg_df = marginals_to_pandas(a_stats)
    posteriors = analyzer.get_equal_weighted_posterior()
    with h5py.File(store_name, 'a') as hdf:
        group = hdf.create_group(group_name)
        # general attributes:
        group.attrs['model_name']     = runner.model_name
        group.attrs['ncomp']          = runner.ncomp
        group.attrs['par_labels']     = runner.par_labels
        group.attrs['std_noise']      = runner.noise
        group.attrs['null_lnZ']       = runner.null_lnZ
        group.attrs['global_lnZ']     = a_stats['global evidence']
        group.attrs['global_lnZ_err'] = a_stats['global evidence error']
        group.attrs['max_loglike']    = bestfit['log_likelihood']
        # datasets:
        group.create_dataset('map_params', data=np.array(bestfit['parameters']))
        group.create_dataset('posteriors', data=posteriors)
        group.create_dataset('marginals', data=marg_df.values)


if __name__ == '__main__':
    pass


