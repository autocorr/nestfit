#!/usr/bin/env python3
"""
Test the evidence values for different number of components when varying the
noise level.
"""

import operator
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib import pyplot as plt


plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', direction='out', top=True)
plt.rc('ytick', direction='out', right=True)


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


