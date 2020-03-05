#!/usr/bin/env python3
"""
Plotting module for visualizing spectra and fit results.
"""

from pathlib import Path

import numpy as np
from scipy import special
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mayavi import mlab as mvi

import getdist
from getdist import plots as gd_plt
import pyspeckit
from astropy.wcs import WCS

from nestfit import (get_par_names, TEX_LABELS, TEX_LABELS_NU)
from nestfit.synth_spectra import (SyntheticSpectrum, get_test_spectra)
from nestfit.wrapped import (amm11_predict, amm22_predict)


plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', direction='out', top=True)
plt.rc('ytick', direction='out', right=True)


CLR_CMAP = plt.cm.Spectral_r
CLR_CMAP.set_bad('0.5', 1.0)
HOT_CMAP = plt.cm.afmhot
HOT_CMAP.set_bad('0.5', 1.0)
RDB_CMAP = plt.cm.RdBu
RDB_CMAP.set_bad('0.5', 1.0)
VIR_CMAP = plt.cm.viridis
VIR_CMAP.set_bad('0.5', 1.0)

_cmap_list = [(0.5, 0.5, 0.5, 1.0)] + [plt.cm.plasma(i) for i in range(plt.cm.plasma.N)]
NBD_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
        'Discrete Plasma', _cmap_list, len(_cmap_list),
)
NBD_CMAP.set_bad('0.2')


def save_figure(filen):
    exts = ('png', 'pdf')
    for ext in exts:
        path = Path(f'{filen}.{ext}')
        plt.savefig(str(path), dpi=300)
        print(f'-- {ext} saved')
    plt.close('all')
    plt.cla()
    plt.clf()


def add_scaled_colorbar(mappable):
    """
    NOTE The following code was written by Joseph Long and the original may be
    found here:
        https://joseph-long.com/writing/colorbars/
    """
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def plot_synth_spectra(spectra=None):
    if spectra is None:
        spectra = get_test_spectra()
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
    save_figure(outname)


def plot_corner(group, outname='corner'):
    ncomp = group.attrs['ncomp']
    par_labels = TEX_LABELS
    n_params = group.attrs['n_params'] // ncomp
    names = get_par_names()
    post = group['posteriors'][...][:,:-2]  # don't need logL & P
    # Give each model component parameter set its own sampler object so that
    # each can be over-plotted in its own color.
    samples = [
            getdist.MCSamples(
                samples=post[:,ii::ncomp],
                names=names,
                labels=par_labels,
                label=f'Component {ii+1}')
            for ii in range(ncomp)
    ]
    fig = gd_plt.get_subplot_plotter()
    fig.triangle_plot(samples, filled=True,
            line_args=[
                {'lw':2, 'color':'tab:orange'},
                {'lw':2, 'color':'tab:blue'},
                {'lw':2, 'color':'tab:green'}],
    )
    fig.export(f'{outname}.pdf')
    plt.close()


def plot_multicomp_velo_2corr(group, outname='velo_2corr'):
    ncomp = group.attrs['ncomp']
    assert ncomp == 2
    n_params = group.attrs['n_params'] // ncomp
    post = group['posteriors'][...][:,:-2]  # don't need logL & P
    names = get_par_names(ncomp)
    samples = getdist.MCSamples(samples=post, names=names)
    fig = gd_plt.get_subplot_plotter()
    x_names = ['v1', 's1']
    y_names = ['v2', 's2']
    fig.rectangle_plot(x_names, y_names, roots=samples, filled=True)
    fig.export(f'{outname}.pdf')
    plt.close()


def add_discrete_colorbar(ax, orientation='vertical'):
    t_cbar = plt.colorbar(ax)
    t_cbar.ax.clear()
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    ticks  = [0, 1, 2, 3, 4]
    norm  = mpl.colors.BoundaryNorm(bounds, NBD_CMAP.N)
    cbar = mpl.colorbar.ColorbarBase(t_cbar.ax, cmap=NBD_CMAP, norm=norm,
            boundaries=bounds, ticks=ticks, spacing='uniform', orientation=orientation)
    return cbar


def plot_agg_nbest(store, group='/aggregate/independent/nbest_image',
        outname='nbest'):
    data = store.hdf[group]
    wcs = WCS(store.read_header(full=False))
    fig, ax = plt.subplots(figsize=(4, 3.3), subplot_kw={'projection': wcs})
    im = ax.imshow(data, vmin=0, vmax=4, cmap=NBD_CMAP)
    cbar = add_discrete_colorbar(im)
    cbar.set_label(r'$N_\mathrm{best}$')
    ax.set_xlabel(r'$\mathrm{Right\ Ascension\ (J2000)}$')
    ax.set_ylabel(r'$\mathrm{Declination\ (J2000)}$', labelpad=-0.8)
    plt.tight_layout()
    plt.subplots_adjust(left=0.20, right=0.95, bottom=0.15, top=0.95)
    save_figure(f'{outname}')


def plot_map_props(store, group='/aggregate/independent/nbest_MAP_cube',
        outname='map_props'):
    data = store.hdf[group][...]
    n_mod, n_params, _, _ = data.shape
    wcs = WCS(store.read_header(full=False))
    # iterate through properties
    for ii in range(n_params):
        fig, axes = plt.subplots(nrows=1, ncols=n_mod, figsize=(8, 2.8),
                subplot_kw={'projection': wcs})
        vmin = np.nanmin(data[:,ii,:,:])
        vmax = np.nanmax(data[:,ii,:,:])
        for jj, ax in enumerate(axes):
            img = data[jj,ii,:,:]
            im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=CLR_CMAP)
            if not ax.is_first_col():
                ax.tick_params(axis='y', labelleft=False)
        cax = fig.add_axes([0.89, 0.20, 0.015, 0.70])
        cbar = plt.colorbar(im, cax=cax)
        cbar.minorticks_on()
        cbar.set_label(TEX_LABELS[ii])
        axes[0].set_xlabel(r'$\mathrm{Right\ Ascension\ (J2000)}$')
        axes[0].set_ylabel(r'$\mathrm{Declination\ (J2000)}$', labelpad=-0.8)
        plt.tight_layout()
        plt.subplots_adjust(left=0.10, right=0.875, bottom=0.15, top=0.95)
        save_figure(f'{outname}_par{ii}')


def plot_3d_volume(store, outname='volume_field_contour'):
    histdata = store.hdf['/aggregate/independent/post_hists'][...]
    vals = histdata[0,...]
    vmin, vmax = vals.min(), vals.max()
    obj = mvi.contour3d(vals, colormap='inferno', transparent=True)
    mvi.savefig(f'plots/{outname}.pdf')
    mvi.clf()


def plot_specfit(store, stack, pix, n_model=1, outname='specfit'):
    lon_pix, lat_pix = pix
    group = store.hdf[f'/pix/{lon_pix}/{lat_pix}/{n_model}']
    params = group['map_params'][...]
    spectra = stack.get_arrays(*pix)
    freq_dict = pyspeckit.spectrum.models.ammonia.freq_dict
    rest_freqs = (freq_dict['oneone'], freq_dict['twotwo'])
    xarrs = [
            pyspeckit.spectrum.units.SpectroscopicAxis(cube.xarr, unit='Hz',
                refX=rest_freqs[i], refX_unit='Hz',
                velocity_convention='radio')
            for i, cube in enumerate(stack.cubes)
    ]
    synspectra = [SyntheticSpectrum(x, params) for x in xarrs]
    fig = plt.figure(figsize=(4, 5))
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1 = plt.subplot2grid((3, 1), (2, 0))
    axes = (ax0, ax1)
    #fig, axes = plt.subplots(nrows=len(spectra), ncols=1,
    #        sharex=True, sharey=True, figsize=(4, 5))
    for data, xarr, synspec, ax in zip(spectra, xarrs, synspectra, axes):
        varr = synspec.varr
        ax.fill_between(varr, data, np.zeros_like(data), color='yellow',
                edgecolor='none', alpha=0.5)
        ax.plot(varr, data, 'k-', linewidth=0.7, drawstyle='steps-pre')
        ax.plot(varr, synspec.components.T, '-', color='magenta', linewidth=1.0, alpha=0.5)
        ax.plot(varr, synspec.sum_spec, '-', color='red', linewidth=1.0)
        ax.set_xlim(varr.value.min(), varr.value.max())
    ymin = spectra[0].min() * 1.1
    ymax = spectra[0].max() * 1.1
    axes[0].set_ylim(ymin, ymax)
    axes[1].set_ylim(ymin*0.4-0.5, ymax*0.4-0.5)
    axes[1].set_xlabel(r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$')
    axes[1].set_ylabel(r'$T_\mathrm{mb} \ [\mathrm{K}]$')
    axes[0].annotate(r'$\mathrm{NH_3}\, (1,1)$', (0.03, 0.91), xycoords='axes fraction')
    axes[1].annotate(r'$\mathrm{NH_3}\, (2,2)$', (0.03, 0.80), xycoords='axes fraction')
    plt.tight_layout()
    save_figure(f'{outname}_{lon_pix}_{lat_pix}_{n_model}')


def test_wrapped_amm_precision():
    spectra = get_test_spectra()
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
    plt.savefig(Path('cython_test_compare_precision.pdf'))
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
    plt.savefig(Path('partition_function.pdf'))
    plt.close('all')


