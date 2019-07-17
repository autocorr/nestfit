#!/usr/bin/env python3
"""
Gaussian mixture fitting with Nested Sampling.
"""

import operator
from pathlib import Path

import h5py
import numpy as np
import scipy as sp
import pandas as pd
from scipy import (special, stats)
from matplotlib import ticker
from matplotlib import pyplot as plt

import corner
import pyspeckit
import pymultinest
from astropy import units as u

from nestfit.wrapped import (amm11_predict, amm22_predict, PriorTransformer,
        AmmoniaRunner)


plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', direction='out', top=True)
plt.rc('ytick', direction='out', right=True)

ROOT_DIR = Path('/lustre/aoc/users/bsvoboda/temp/nestfit')
DATA_DIR = ROOT_DIR / Path('data')
PLOT_DIR = ROOT_DIR / Path('plots')

TEX_LABELS = [
        r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$',
        r'$T_\mathrm{rot} \ [\mathrm{K}]$',
        r'$T_\mathrm{ex} \ [\mathrm{K}]$',
        r'$\log(N_\mathrm{tot}) \ [\mathrm{cm^{-2}}]$',
        r'$\sigma_\mathrm{v} \ [\mathrm{km\, s^{-1}}]$',
]


def get_par_labels(ncomp):
    return [
            f'{label}{n}'
            for label in ('v', 'Tk', 'Tx', 'N', 's')
            for n in range(1, ncomp+1)
    ]


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


def run_trials_varying_noise(store_name='varnoise'):
    xaxis = np.linspace(-6, 6, 100)
    amp = np.array([0.3, 0.5, 0.4])
    cen = np.array([-1, 0, 3])
    std = np.array([1.5, 1.0, 0.5])
    args = xaxis, amp, cen, std
    # sample noise values log-uniformly from 1 to 100 peak-SNR
    all_noises = 0.75 / np.logspace(0, 2, 100)
    for ii, noise in enumerate(all_noises):
        spec = SyntheticSpectrum(*args, noise=noise, set_seed=False)
        for ncomp in range(1, 5):
            group_name = f'spec_{ii:0>4d}/ncomp_{ncomp}'
            model = GaussianModel(
                    spec.xaxis,
                    spec.sampled_spec,
                    spec.noise,
                    ncomp,
            )
            analyzer = run_nested(spec, model)
            save_run(model, analyzer, group_name, store_name=store_name)


def plot_synth_spectra(spectra=None):
    if spectra is None:
        spectra = test_spectra()
    fig = plt.figure(figsize=(4, 6))
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1 = plt.subplot2grid((3, 1), (2, 0))
    ymax = spectra[1].sampled_spec.max() * 1.1
    ymin = -3 * spectra[1].noise
    ax0.set_ylim(2*ymin, 2*ymax)
    ax1.set_ylim(ymin, ymax)
    axes = [ax0, ax1]
    for spec, ax in zip(spectra, axes):
        ax.plot(spec.varr, spec.sampled_spec, color='black',
                drawstyle='steps-mid', linewidth=0.7)
        #ax.fill_between(spec.varr, spec.sampled_spec, step='mid',
        #        edgecolor='none', facecolor='yellow', alpha=0.8)
        #ax.plot(spec.varr, spec.components.T, '-', color='magenta',
        #        linewidth=0.7)
        ax.set_xlim(spec.varr.value.min(), spec.varr.value.max())
    labels = [r'$\mathrm{NH_3}\, (1,1)$', r'$\mathrm{NH_3}\, (2,2)$']
    for label, ax in zip(labels, axes):
        ax.annotate(label, xy=(0.05, 0.85), xycoords='axes fraction')
    ax.set_xlim(spec.varr.min().value, spec.varr.max().value)
    ax.set_ylabel(r'$T_\mathrm{b}$')
    ax.set_xlabel(r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$')
    plt.tight_layout()
    plt.savefig(f'plots/test_synthetic_ammonia_spectra.pdf')
    plt.close('all')


def plot_spec_compare(synspec, analyzer, outname='test'):
    n = synspec[0].ncomp
    varr = synspec[0].varr
    fig = plt.figure(figsize=(4, 6))
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1 = plt.subplot2grid((3, 1), (2, 0))
    ymax = synspec[1].sampled_spec.max() * 1.1
    ymin = -3 * synspec[1].noise
    ax0.set_ylim(2*ymin, 2*ymax)
    ax1.set_ylim(ymin, ymax)
    axes = [ax0, ax1]
    ## Comparison of the synthetic spectrum and a draw from the posteriors
    # observed data
    for spec, ax in zip(synspec, axes):
        ax.step(varr, spec.sampled_spec, color='black', linewidth=0.7)
        ax.step(varr, spec.sampled_spec, color='black', linewidth=0.7)
        # plot a sub-sample of spectra
        posteriors = analyzer.get_equal_weighted_posterior()[:,:-1]
        posteriors = posteriors[::len(posteriors)//30]
        samples = [
            SyntheticSpectrum(spec.xarr, row) for row in posteriors
        ]
        for sampled_spec in samples:
            #ax.plot(varr, sampled_spec.components.T, '-', color='red',
            #        alpha=0.1)
            ax.plot(varr, sampled_spec.sum_spec, '-', color='red',
                    alpha=0.1, linewidth=0.5)
        # individual true components
        #ax.plot(varr, spec.components.T, '-', color='magenta', linewidth=0.7)
        # best fit spectrum
        best_pars = np.array(analyzer.get_best_fit()['parameters'])
        best_spec = SyntheticSpectrum(varr, best_pars)
        #ax.plot(varr, spec.sum_spec, '-', color='dogerblue', linewidth=0.7)
        ax.set_xlim(spec.varr.value.min(), spec.varr.value.max())
    ax.set_xlabel(r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$')
    ax.set_ylabel(r'$T_\mathrm{b} \ [\mathrm{K}]$')
    ## Comparison of the "true" residuals and the residuals from the best-fit
    # use error function to represent true noise distribution and use the true
    # noise drawn in the synthetic spectrum.
    #bins = np.linspace(-0.1, 0.1, 100)
    #errdist = 0.5 * special.erf(bins/runner.noise) + 0.5
    #ax1.plot(bins, errdist, '-', color='0.5')
    #ax1.hist(synspec.noise_spec, bins=bins, cumulative=True, density=True,
    #        histtype='step', color='black',)
    #ax1.hist(runner.ydata-best_spec.sum_spec, bins=bins, cumulative=True, density=True,
    #        histtype='step', color='red')
    #ax1.set_xlabel(r'$T_\mathrm{b} \ [\mathrm{K}]$')
    #ax1.set_ylabel(r'$\mathrm{Residual\ CDF}$')
    # save figure
    plt.tight_layout()
    plt.savefig(f'plots/{outname}.pdf')
    plt.close('all')


def plot_corner(synspec, analyzer, show_truths=False, outname='test_corner'):
    par_labels = get_par_labels(analyzer.n_params // 5)
    truths = synspec.params if show_truths else None
    plt.rc('font', size=12, family='serif')
    posteriors = analyzer.get_equal_weighted_posterior()[:,:-1]
    fig = corner.corner(posteriors, truths=truths,
            labels=par_labels, label_kwargs={'fontsize': 14},
            show_titles=True, title_kwargs={'fontsize': 14})
    # save figure
    plt.savefig(f'plots/{outname}.pdf')
    plt.close('all')
    plt.rc('font', size=10, family='serif')


def read_varnoise_summary(store_file='varnoise'):
    if not store_file.endswith('.hdf5'):
        store_file += '.hdf5'
    df = pd.DataFrame()
    nchan = 100
    with h5py.File(store_file, 'r') as hdf:
        for run_group in hdf.values():
            spec_name = run_group.name.lstrip('/')
            for fit_group in run_group.values():
                ncomp = fit_group.attrs['ncomp']
                kpar = 3 * ncomp
                lnZ = fit_group.attrs['global_lnZ']
                lnZ_err = fit_group.attrs['global_lnZ_err']
                maxL = fit_group.attrs['max_loglike']
                bic = np.log(nchan) * kpar - 2 * maxL
                aic = 2 * kpar - 2 * maxL
                aicc = aic + (2 * kpar**2 + 2 * kpar) / (nchan - kpar - 1)
                df.loc[spec_name, f'lnZ{ncomp}'] = lnZ
                df.loc[spec_name, f'lnZ{ncomp}_err'] = lnZ_err
                df.loc[spec_name, f'maxL{ncomp}'] = maxL
                df.loc[spec_name, f'BIC{ncomp}'] = bic
                df.loc[spec_name, f'AIC{ncomp}'] = aic
                df.loc[spec_name, f'AICc{ncomp}'] = aicc
            lnZ0 = fit_group.attrs['null_lnZ']
            df.loc[spec_name, 'lnZ0']  = lnZ0
            df.loc[spec_name, 'BIC0']  = -2 * lnZ0
            df.loc[spec_name, 'AIC0']  = -2 * lnZ0
            df.loc[spec_name, 'AICc0'] = -2 * lnZ0 + 2 / (nchan - 1)
            df.loc[spec_name, 'noise'] = fit_group.attrs['std_noise']
    return df


def plot_varnoise_evidence_noise(df):
    snr = 0.75 / df.noise
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hlines([0], snr.min(), snr.max(), color='0.5', linewidth=0.7,
            linestyle='dotted')
    label1 = r'$\mathrm{ln}(\mathcal{Z}_1/\mathcal{Z}_0)$'
    label2 = r'$\mathrm{ln}(\mathcal{Z}_2/\mathcal{Z}_1)$'
    label3 = r'$\mathrm{ln}(\mathcal{Z}_3/\mathcal{Z}_2)$'
    label4 = r'$\mathrm{ln}(\mathcal{Z}_4/\mathcal{Z}_3)$'
    line_kwargs = {'drawstyle': 'steps-mid'}
    ax.plot(snr, df.lnZ1-df.lnZ0, label=label1, **line_kwargs)
    ax.plot(snr, df.lnZ2-df.lnZ1, label=label2, **line_kwargs)
    ax.plot(snr, df.lnZ3-df.lnZ2, label=label3, **line_kwargs)
    ax.plot(snr, df.lnZ4-df.lnZ3, label=label4, **line_kwargs)
    ax.hlines([-16.1, 16.1], snr.min(), snr.max(), color='red', linewidth=0.7,
            linestyle='dotted')
    ax.legend(loc='lower left', ncol=2, fancybox=False, fontsize='x-small')
    ax.set_xscale('log')
    ax.set_xlim(snr.min(), snr.max())
    ax.set_ylim(-50, 100)
    ax.set_xlabel(r'$\mathrm{max}(I_\nu) / \sigma_\mathrm{rms}$')
    ax.set_ylabel(r'$\mathrm{ln}(\mathcal{Z}_{i} / \mathcal{Z}_{i-1})$')
    plt.tight_layout()
    plt.savefig(PLOT_DIR/Path('evidence_by_noise.pdf'))
    plt.close('all')


def plot_varnoise_aic_bic_noise(df):
    snr = 0.75 / df.noise
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hlines([0], snr.min(), snr.max(), color='0.5', linewidth=0.7,
            linestyle='dotted')
    label1 = r'$\Delta \mathrm{BIC}(1-0)$'
    label2 = r'$\Delta \mathrm{BIC}(2-1)$'
    label3 = r'$\Delta \mathrm{BIC}(3-2)$'
    label4 = r'$\Delta \mathrm{BIC}(4-3)$'
    line_kwargs = {'drawstyle': 'steps-mid'}
    #ax.plot(snr, df.AICc1-df.AICc0, label=label1, **line_kwargs)
    #ax.plot(snr, df.AICc2-df.AICc1, label=label2, **line_kwargs)
    #ax.plot(snr, df.AICc3-df.AICc2, label=label3, **line_kwargs)
    #ax.plot(snr, df.AICc4-df.AICc3, label=label4, **line_kwargs)
    ax.plot(snr, df.BIC1-df.BIC0, label=label1, **line_kwargs)
    ax.plot(snr, df.BIC2-df.BIC1, label=label2, **line_kwargs)
    ax.plot(snr, df.BIC3-df.BIC2, label=label3, **line_kwargs)
    ax.plot(snr, df.BIC4-df.BIC3, label=label4, **line_kwargs)
    ax.hlines([-16.1], snr.min(), snr.max(), color='red', linewidth=0.7,
            linestyle='dotted')
    ax.legend(loc='upper left', ncol=2, fancybox=False, fontsize='x-small')
    ax.set_xscale('log')
    ax.set_xlim(snr.min(), snr.max())
    ax.set_ylim(-100, 50)
    ax.set_xlabel(r'$\mathrm{max}(I_\nu) / \sigma_\mathrm{rms}$')
    ax.set_ylabel(r'$\Delta \mathrm{BIC}$')
    plt.tight_layout()
    plt.savefig(PLOT_DIR/Path('aic_bic_by_noise.pdf'))
    plt.close('all')


def plot_varnoise_metrics_compare(df):
    df = df.copy()
    df['dlnZ10'] = df.lnZ1 - df.lnZ0
    df['dlnZ21'] = df.lnZ2 - df.lnZ1
    df['dlnZ32'] = df.lnZ3 - df.lnZ2
    df['dlnZ43'] = df.lnZ4 - df.lnZ3
    df['dBIC10'] = df.BIC1 - df.BIC0 - np.log(100) * 3
    df['dBIC21'] = df.BIC2 - df.BIC1 - np.log(100) * 3
    df['dBIC32'] = df.BIC3 - df.BIC2 - np.log(100) * 3
    df['dBIC43'] = df.BIC4 - df.BIC3 - np.log(100) * 3
    snr = 0.75 / df.noise
    fig, ax = plt.subplots(figsize=(4, 3))
    label1 = r'$(1-0)$'
    label2 = r'$(2-1)$'
    label3 = r'$(3-2)$'
    label4 = r'$(4-3)$'
    plot_kwargs = {'marker': 'o', 'markersize': 3, 'linestyle': 'none'}
    ax.plot(snr, (df.dBIC10-df.dlnZ10)/(df.dBIC10+df.dlnZ10),
            label=label1, **plot_kwargs)
    ax.plot(snr, (df.dBIC21-df.dlnZ21)/(df.dBIC21+df.dlnZ21),
            label=label2, **plot_kwargs)
    ax.plot(snr, (df.dBIC32-df.dlnZ32)/(df.dBIC32+df.dlnZ32),
            label=label3, **plot_kwargs)
    ax.plot(snr, (df.dBIC43-df.dlnZ43)/(df.dBIC43+df.dlnZ43),
            label=label4, **plot_kwargs)
    ax.legend(loc='lower right', ncol=2, fancybox=False, fontsize='x-small')
    ax.set_xscale('log')
    ax.set_xlim(snr.min(), snr.max())
    ax.set_ylim(-5, 5)
    ax.set_xlabel(r'$\mathrm{max}(I_\nu) / \sigma_\mathrm{rms}$')
    ax.set_ylabel(r'$\left[2\Delta \mathcal{L}_\mathrm{max} - \Delta \mathrm{ln}(\mathcal{Z})\right] / \left[2\Delta \mathcal{L}_\mathrm{max} + \Delta \mathrm{ln}(\mathcal{Z})\right]$')
    plt.tight_layout()
    plt.savefig(PLOT_DIR/Path('diff_evidence_bic_by_noise.pdf'))
    plt.close('all')


def plot_varnoise_preferred_model(df):
    df = df.copy()
    df['snr'] = 0.75 / df.noise
    lnZ_cols = [f'lnZ{n}'  for n in range(5)]
    bic_cols = [f'BIC{n}'  for n in range(5)]
    aic_cols = [f'AICc{n}' for n in range(5)]
    # too complicated but, oh well, it's here and it works
    def set_nbest(cols, thresh, outcol='nbest', comp_op='<'):
        op = operator.lt if comp_op == '<' else operator.gt
        for ix in df.index:
            row = df.loc[ix, cols]
            for ii in range(4, 0, -1):
                if op(row[cols[ii]] - row[cols[ii-1]], thresh):
                    break
            else:
                ii = 0
            df.loc[ix, outcol] = ii
    set_nbest(lnZ_cols,  16.1, outcol='lnZ_nbest', comp_op='>')
    set_nbest(bic_cols, -16.1, outcol='bic_nbest', comp_op='<')
    set_nbest(aic_cols, -16.1, outcol='aic_nbest', comp_op='<')
    fig, ax = plt.subplots(figsize=(4, 2))
    plot_kwargs = {'marker': 'o', 'markersize': 1.4, 'linestyle': 'none'}
    ax.plot(df.snr, df.aic_nbest+0.2, color='dodgerblue',
            label=r'$\mathrm{AICc}$', **plot_kwargs)
    ax.plot(df.snr, df.bic_nbest+0.1, color='red',
            label=r'$\mathrm{BIC}$', **plot_kwargs)
    ax.plot(df.snr, df.lnZ_nbest, color='black',
            label=r'$\mathrm{ln}(\mathcal{Z})$', **plot_kwargs)
    ax.legend(loc='upper left', ncol=1, fancybox=False, fontsize='x-small')
    ax.hlines([0], df.snr.min(), df.snr.max(), color='0.5', linestyle='dotted')
    ax.set_xscale('log')
    ax.set_yticks(range(5))
    ax.set_xlim(df.snr.min(), df.snr.max())
    ax.set_ylim(-0.2, 4.2)
    ax.set_xlabel(r'$\mathrm{max}(I_\nu) / \sigma_\mathrm{rms}$')
    ax.set_ylabel(r'$N_\mathrm{best}$')
    plt.tight_layout()
    plt.savefig(PLOT_DIR/Path('preferred_model.pdf'))
    plt.close('all')


def plot_varnoise_spec_examples(store_name='varnoise'):
    if not store_name.endswith('.hdf5'):
        store_name += '.hdf5'
    tspec = test_spectrum()
    def test_spectrum_with_noise(noise):
        return SyntheticSpectrum(tspec.xaxis, tspec.amp, tspec.cen, tspec.std,
                noise=noise, set_seed=False)
    def parse_spectrum(ncomp, params, noise):
        ncomp = 1 if ncomp == 0 else ncomp
        return SyntheticSpectrum(tspec.xaxis, params[:ncomp],
                params[ncomp:2*ncomp], params[2*ncomp:3*ncomp], noise=noise,
                set_seed=False)
    fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True, sharey=True,
            figsize=(4, 6))
    xaxis = tspec.xaxis.flatten()
    with h5py.File(store_name, 'r') as hdf:
        spec_ix = (5, 15, 50, 80)
        for ix, n, ax in zip(spec_ix, range(4), axes):
            group_name = f'/spec_{ix:0>4d}/ncomp_{1 if n==0 else n}'
            noise = hdf[group_name].attrs['std_noise']
            pars  = hdf[group_name+'/map_params']
            spec  = parse_spectrum(n, pars, noise)
            nspec = test_spectrum_with_noise(noise)
            sigmaones = nspec.noise * np.ones(xaxis.shape)
            ax.fill_between(xaxis, -sigmaones, sigmaones, color='yellow',
                    edgecolor='none', alpha=0.5)
            label = r'$N_\mathrm{best} = ' + str(n) + '$'
            ax.annotate(label, (0.05, 0.8), xycoords='axes fraction')
            ax.plot(xaxis, nspec.sampled_spec, color='black',
                    drawstyle='steps-mid')
            ax.plot(xaxis, tspec.components.T, color='magenta', linewidth=0.75)
            ax.plot(xaxis,
                    np.zeros(xaxis.shape) if n == 0 else spec.components.T,
                    color='cyan', linewidth=1.0)
            ax.plot(xaxis,
                    np.zeros(xaxis.shape) if n == 0 else spec.sum_spec,
                    color='dodgerblue', linewidth=0.75)
    ax.set_ylim(-0.75, 2.0)
    ax.set_xlabel(r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$')
    ax.set_ylabel(r'$T_\mathrm{b} \ [\mathrm{K}]$')
    plt.tight_layout()
    plt.savefig(PLOT_DIR/Path('MAP_best_for_noise.pdf'))
    plt.close('all')


def test_wrapped_amm_precision():
    spectra = test_spectra()
    funcs = (amm11_predict, amm22_predict)
    # plotting
    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(4, 3))
    for syn, predict, ax in zip(spectra, funcs, axes):
        xarr = syn.xarr.value.copy()  # xarr is read-only
        data = np.empty_like(xarr)
        params = syn.params
        predict(xarr, data, params)
        diff = np.log10(np.abs((data-syn.sum_spec)))
        diff[diff < -12] = np.nan
        print(':: max log10(diff) =', np.nanmax(diff))
        ax.plot(syn.varr, diff,
                'k-', drawstyle='steps-mid', linewidth=0.7)
    ax.set_xlim(-30, 30)
    ax.set_xlabel(r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$')
    ax.set_ylabel(r'$\log_\mathrm{10}\left( |\Delta T_\mathrm{b}| \right) \ [\mathrm{K}]$')
    plt.tight_layout()
    plt.savefig(PLOT_DIR/Path('cython_test_compare_precision.pdf'))
    plt.close('all')


def test_poly_partition_function():
    brot = 298117.06e6
    crot = 186726.36e6
    h    = 6.62607004e-27  # erg s
    kb   = 1.38064852e-16  # erg/K
    nlevs = 51
    trot = np.linspace(0.0, 50.0, 1e3)  # K
    Qtot = np.zeros_like(trot)
    for j in range(0, nlevs+1):
        nuc_spin = 2 if j % 3 == 0 else 1
        Qtot += (
            (2 * j + 1) * nuc_spin * np.exp(-h * (brot * j * (j + 1)
            + (crot - brot) * j**2) / (kb * trot))
        )
    # plotting
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(trot, Qtot, 'k-')
    ax.set_xlim(0, 50)
    ax.set_ylim(0,  6)
    ax.set_xlabel(r'$T_\mathrm{rot} \ [\mathrm{K}]$')
    ax.set_ylabel(r'$Q_\mathrm{tot}$')
    plt.tight_layout()
    plt.savefig(PLOT_DIR/Path('partition_function.pdf'))
    plt.close('all')


if __name__ == '__main__':
    test_wrapped_amm_precision()
    #_ = test_nested(ncomp=1)


