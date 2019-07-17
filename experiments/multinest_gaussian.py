#!/usr/bin/env python3
"""
Gaussian mixture fitting with Nested Sampling. This module was tested in the
main `nestfit` repo on bare arrays and Gaussian components -- without a
spectral axis, units, or other necessary complications.

The `.wrapped` references a Cython implementation of the Gaussian model class.
"""

import ctypes
import operator
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import (special, stats)
from matplotlib import ticker
from matplotlib import pyplot as plt

import corner
import pymultinest

from .wrapped import CGaussianModel


plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', direction='out', top=True)
plt.rc('ytick', direction='out', right=True)

ROOT_DIR = Path('/lustre/aoc/users/bsvoboda/temp/nestfit')
DATA_DIR = ROOT_DIR / Path('data')
PLOT_DIR = ROOT_DIR / Path('plots')


class SyntheticSpectrum:
    def __init__(self, xaxis, amp, cen, std, noise=0.03, set_seed=False):
        """
        Construct a mixture of Gaussians expressed as:
            f(x) = A * exp(-(x - c)^2 / (2 * s^2))
        for "A" amplitude, "c" centroid, and "s" standard deviation.

        Parameters
        ----------
        xaxis : np.ndarray
        amp : np.ndarray
            Array of Gaussian amplitudes
        cen : np.ndarray
            Array of Gaussian centroid positions
        std : np.ndarray
            Array of Guassian standard deviations
        noise : float, default=0.03
            Noise standard deviation
        set_seed : bool, default=False
            If `True` will use a default seed of 5 for the np.random module.
        """
        if set_seed:
            np.random.seed(5)
        else:
            np.random.seed()
        self.xaxis = xaxis.reshape(-1, 1)
        self.ncomp = len(amp)
        self.size  = self.xaxis.shape[0]
        self.amp = amp
        self.cen = cen
        self.std = std
        self.truths = np.concatenate([amp, cen, std])
        self.noise = noise
        self.components = self.profile().T
        self.sum_spec = self.components.sum(axis=0)
        self.noise_spec = np.random.normal(scale=self.noise, size=self.size)
        self.sampled_spec = self.sum_spec + self.noise_spec

    def profile(self):
        return self.amp * np.exp(-(self.xaxis - self.cen)**2 / (2 * self.std**2))

    def resample_spectrum(self, noise=None):
        if noise is not None:
            self.noise = noise
        noise_spec = np.random.normal(scale=self.noise, size=self.size)
        self.noise_spec = noise_spec
        self.sampled_spec = self.sum_spec + self.noise_spec


def test_spectrum():
    return SyntheticSpectrum(
            np.linspace(-6, 6, 100),
            amp=np.array([0.3, 0.5, 0.4]),
            cen=np.array([-1, 0, 3]),
            std=np.array([1.5, 1.0, 0.5]),
            noise=0.03,
            set_seed=True,
    )


class GaussianModel:
    model_name = 'gaussian'

    def __init__(self, xaxis, ydata, noise, ncomp):
        self.xaxis = xaxis.reshape(-1, 1)
        self.size  = xaxis.shape[0]
        self.ydata = ydata
        self.noise = noise
        self.ncomp = ncomp
        self.n_params = 3 * ncomp
        self.lnpin = -self.size / 2 * np.log(2 * np.pi * noise**2)
        self.null_lnZ = self.lnpin - np.sum(ydata**2) / (2 * self.noise**2)
        #self.array_type = np.ctypeslib.ndpointer(
        #        ctypes.c_double, 1, (self.n_params,), 'C_CONTIGUOUS')

    @property
    def par_labels(self):
        comps = range(1, self.ncomp+1)
        return [
                f'{label}{n}'
                for label in ('a', 'c', 's')
                for n in comps
        ]

    def loglikelihood(self, theta, ndim, nparams):
        n = self.ncomp
        #atheta = ctypes.cast(theta, self.array_type).contents
        atheta = np.ctypeslib.as_array(theta, shape=(self.n_params,))
        amp = atheta[0  :  n]
        cen = atheta[  n:2*n]
        std = atheta[2*n:3*n]
        ymodel = np.sum(
                amp * np.exp(-(self.xaxis - cen)**2 / (2 * std**2)),
                axis=1,
        )
        difsqsum = np.sum((self.ydata - ymodel)**2)
        lnL = self.lnpin - difsqsum / (2 * self.noise**2)
        return lnL

    def prior_transform(self, utheta, ndim, nparams):
        n = self.ncomp
        # amplitude -- uniform [0.06, 1.00]
        for i in range(0, n):
            utheta[i] = 0.94 * utheta[i] + 0.06
        # centroid velocity -- uniform [-5.00, 5.00]
        #  but enforce ordering from left-to-right for the peaks to sort
        #  and limit multi-modality in posteriors
        vmin, vmax = -5.0, 5.0
        for i in range(n, 2*n):
            v = (vmax - vmin) * utheta[i] + vmin
            utheta[i] = vmin = v
        # standard deviation -- uniform [0.30, 3.00]
        for i in range(2*n, 3*n):
            utheta[i] = 2.7 * utheta[i] + 0.30
        return utheta  # XXX


def run_nested(spec, model, basename='run/test_run'):
    pymultinest.run(
            model.loglikelihood,
            model.prior_transform,
            model.n_params,
            outputfiles_basename=basename,
            resume=False,
            verbose=True,
            evidence_tolerance=0.3,
            n_live_points=400,
            sampling_efficiency=0.3,
            n_iter_before_update=2000,
    )
    analyzer = pymultinest.Analyzer(
            outputfiles_basename=basename,
            n_params=model.n_params,
    )
    lnZ = analyzer.get_stats()['global evidence']
    print(':: Evidence Z:', lnZ/np.log(10))
    return analyzer


def test_nested(ncomp=3):
    spec = test_spectrum()
    model = GaussianModel(
            spec.xaxis,
            spec.sampled_spec,
            spec.noise,
            ncomp,
    )
    analyzer = run_nested(spec, model)
    return spec, model, analyzer


def test_nested_cython(ncomp=3):
    spec = test_spectrum()
    model = CGaussianModel(
            spec.xaxis.flatten(),
            spec.sampled_spec,
            spec.noise,
            ncomp,
    )
    analyzer = run_nested(spec, model)
    return spec, model, analyzer


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


def save_run(model, analyzer, group_name, store_name='nestfit'):
    if not store_name.endswith('.hdf5'):
        store_name += '.hdf5'
    a_stats = analyzer.get_stats()
    bestfit = analyzer.get_best_fit()
    marg_df = marginals_to_pandas(a_stats)
    posteriors = analyzer.get_equal_weighted_posterior()
    with h5py.File(store_name, 'a') as hdf:
        group = hdf.create_group(group_name)
        # general attributes:
        group.attrs['model_name']     = model.model_name
        group.attrs['ncomp']          = model.ncomp
        group.attrs['par_labels']     = model.par_labels
        group.attrs['std_noise']      = model.noise
        group.attrs['null_lnZ']       = model.null_lnZ
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


def plot_spec_compare(synspec, model, analyzer, outname='test'):
    n = model.ncomp
    xaxis = model.xaxis
    fig = plt.figure(figsize=(4, 6))
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1 = plt.subplot2grid((3, 1), (2, 0))
    ## Comparison of the synthetic spectrum and a draw from the posteriors
    # observed data
    ax0.step(xaxis, model.ydata, color='black')
    # plot a sub-sample of spectra
    posteriors = analyzer.get_equal_weighted_posterior()[::100,:-1]
    spectra = [
        SyntheticSpectrum(xaxis, row[:n], row[n:2*n], row[2*n:3*n])
        for row in posteriors
    ]
    for spec in spectra:
        ax0.plot(xaxis, spec.components.T, '-', color='red',
                alpha=0.1)
        ax0.plot(xaxis, spec.sum_spec, '-', color='cyan',
                alpha=0.1)
    # individual true components
    ax0.plot(xaxis, synspec.components.T, '-', color='magenta', linewidth=0.7)
    # best fit spectrum
    best_pars = np.array(analyzer.get_best_fit()['parameters'])
    best_spec = SyntheticSpectrum(xaxis,
            best_pars[:n], best_pars[n:2*n], best_pars[2*n:3*n],
    )
    ax0.plot(xaxis, spec.sum_spec, '-', color='dodgerblue', linewidth=0.7)
    ax0.set_xlabel(r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$')
    ax0.set_ylabel(r'$T_\mathrm{b} \ [\mathrm{K}]$')
    ## Comparison of the "true" residuals and the residuals from the best-fit
    # use error function to represent true noise distribution and use the true
    # noise drawn in the synthetic spectrum.
    bins = np.linspace(-0.1, 0.1, 100)
    errdist = 0.5 * special.erf(bins/model.noise) + 0.5
    ax1.plot(bins, errdist, '-', color='0.5')
    ax1.hist(synspec.noise_spec, bins=bins, cumulative=True, density=True,
            histtype='step', color='black',)
    ax1.hist(model.ydata-best_spec.sum_spec, bins=bins, cumulative=True, density=True,
            histtype='step', color='red')
    ax1.set_xlabel(r'$T_\mathrm{b} \ [\mathrm{K}]$')
    ax1.set_ylabel(r'$\mathrm{Residual\ CDF}$')
    # save figure
    plt.tight_layout()
    plt.savefig(f'plots/{outname}.pdf')
    plt.close('all')


def plot_corner(synspec, model, analyzer, show_truths=False, outname='test_corner'):
    truths = synspec.truths if show_truths else None
    plt.rc('font', size=12, family='serif')
    posteriors = analyzer.get_equal_weighted_posterior()[:,:-1]
    fig = corner.corner(posteriors, truths=truths,
            labels=model.par_labels, label_kwargs={'fontsize': 14},
            show_titles=True, title_kwargs={'fontsize': 14})
    # save figure
    plt.savefig(f'plots/{outname}.pdf')
    plt.close('all')
    plt.rc('font', size=10, family='serif')


def read_varnoise_summary(store_file='varnoise'):
    # FIXME it is probably easier and more elegant to interop between pandas
    # and HDF5 using pytables.
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


