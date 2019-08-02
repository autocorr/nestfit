#!/usr/bin/env python3
"""
Plotting module for visualizing spectra and fit results.
"""

from pathlib import Path

import numpy as np
from scipy import special
from matplotlib import ticker
from matplotlib import pyplot as plt

import corner

from nestfit import (get_par_labels, PLOT_DIR)
from nestfit.main import test_spectra
from nestfit.wrapped import (amm11_predict, amm22_predict)


plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', direction='out', top=True)
plt.rc('ytick', direction='out', right=True)


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


