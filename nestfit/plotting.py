#!/usr/bin/env python3

import itertools
from pathlib import Path

import numpy as np
from scipy import special
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import (patheffects, animation, patches)
from mpl_toolkits.axes_grid1 import make_axes_locatable

import getdist
from getdist import plots as gd_plt
import pyspeckit
from astropy.wcs import WCS

from nestfit.main import take_by_components
from nestfit.synth_spectra import (SyntheticSpectrum, get_test_spectra)
from nestfit.models.ammonia import (AmmoniaSpectrum, amm_predict,
        get_par_names, TEX_LABELS, TEX_LABELS_NU)


plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', direction='out', top=True)
plt.rc('ytick', direction='out', right=True)


CLR_CMAP = plt.cm.magma
CLR_CMAP.set_bad('0.3', 1.0)
RNB_CMAP = plt.cm.Spectral_r
RNB_CMAP.set_bad('0.3', 1.0)
HOT_CMAP = plt.cm.afmhot
HOT_CMAP.set_bad('0.3', 1.0)
STR_CMAP = plt.cm.gist_stern
STR_CMAP.set_bad('0.3', 1.0)
RDB_CMAP = plt.cm.RdBu
RDB_CMAP.set_bad('0.3', 1.0)

_cmap_list = [(0.5, 0.5, 0.5, 1.0)] + [plt.cm.plasma(i) for i in range(plt.cm.plasma.N)]
NBD_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
        'Discrete Plasma', _cmap_list, len(_cmap_list),
)
NBD_CMAP.set_bad('0.2')


##############################################################################
#                        Utilities and Plotting API
##############################################################################

def get_amm_psk_xarrs(stack):
    freq_dict = pyspeckit.spectrum.models.ammonia.freq_dict
    rest_freqs = (freq_dict['oneone'], freq_dict['twotwo'])
    varrs = [
            pyspeckit.spectrum.units.SpectroscopicAxis(cube.xarr, unit='Hz',
                refX=rest_freqs[i], refX_unit='Hz',
                velocity_convention='radio')
            for i, cube in enumerate(stack.cubes)
    ]
    return varrs


def save_figure(filen, dpi=300):
    exts = ('png', 'pdf')
    for ext in exts:
        path = Path(f'{filen}.{ext}')
        plt.savefig(str(path), dpi=dpi)
        print(f'-- {ext} saved')
    plt.close('all')
    plt.cla()
    plt.clf()


def add_discrete_colorbar(ax, vmin=0, vmax=2, orientation='vertical'):
    t_cbar = plt.colorbar(ax, fraction=0.046, pad=0.04)
    t_cbar.ax.clear()
    ticks = np.arange(vmin, vmax+1)
    bounds = np.arange(vmin, vmax+2) - 0.5
    norm  = mpl.colors.BoundaryNorm(bounds, NBD_CMAP.N)
    cbar = mpl.colorbar.ColorbarBase(t_cbar.ax, cmap=NBD_CMAP, norm=norm,
            boundaries=bounds, ticks=ticks, spacing='uniform', orientation=orientation)
    return cbar


class StorePlotter:
    lon_label = r'$\mathrm{Right\ Ascension\ (J2000)}$'
    lat_label = r'$\mathrm{Declination\ (J2000)}$'
    cbar_width = 0.7  # inch
    inch_per_pix = 1.8e-2
    edge_pads = (0.2, 0.1, 0.2, 0.1)  # (l, r, b, t) inch
    adjust_kwargs = {'left': 0.12, 'right': 0.92, 'bottom': 0.15, 'top': 0.95}

    def __init__(self, store, plot_dir=''):
        """
        Parameters
        ----------
        store : HdfStore
        """
        self.store = store
        self.plot_dir = Path(plot_dir)
        self.header = store.read_header(full=True)
        self.wcs = WCS(store.read_header(full=False))
        self.ncomp_max = store.hdf.attrs['n_max_components']

    @property
    def shape(self):
        return self.header['NAXIS1'], self.header['NAXIS2']

    @property
    def pixel_scale(self):
        scale1 = abs(self.header['CDELT1'])
        scale2 = abs(self.header['CDELT2'])
        return scale1, scale2

    def offset_corner_pos(self, offset_frac, loc='upper left'):
        assert loc in ('upper right', 'upper left', 'lower left', 'lower right')
        x, y = self.shape
        y -= 1
        pix_sep = x * offset_frac
        if loc == 'lower left':
            cx = pix_sep
            cy = pix_sep
        elif loc == 'lower right':
            cx = x - pix_sep
            cy = pix_sep
        elif loc == 'upper right':
            cx = x - pix_sep
            cy = y - pix_sep
        elif loc == 'upper left':
            cx = pix_sep
            cy = y - pix_sep
        else:
            raise ValueError(f'Invalid loc: "{loc}"')
        return cx, cy

    def imshow_discrete(self, ax, data, vmin=0, vmax=4):
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=NBD_CMAP)
        cbar = add_discrete_colorbar(im, vmin=vmin, vmax=vmax)
        return im, cbar

    def set_lon_label(self, ax, labelpad=0):
        ax.set_xlabel(self.lon_label)

    def set_lat_label(self, ax, labelpad=-0.8):
        ax.set_ylabel(self.lat_label, labelpad=labelpad)

    def set_labels(self, ax):
        self.set_lon_label(ax)
        self.set_lat_label(ax)

    def set_corner_label(self, ax, label, offset_frac=0.03, loc='lower left'):
        if loc == 'lower left':
            ha, va = 'left', 'bottom'
        elif loc == 'lower right':
            ha, va = 'right', 'bottom'
        elif loc == 'upper right':
            ha, va = 'right', 'top'
        elif loc == 'upper left':
            ha, va = 'left', 'top'
        else:
            raise ValueError(f'Invalid loc: "{loc}"')
        xy = self.offset_corner_pos(offset_frac, loc=loc)
        txt = ax.annotate(label, xy=xy, xycoords='data', fontsize=10,
                horizontalalignment=ha, verticalalignment=va)
        txt.set_path_effects([patheffects.withStroke(linewidth=4.5, foreground='w')])
        return txt

    def axesfrac_to_pixels(self, xy):
        shape = self.shape
        return xy[0]*shape[0], xy[1]*shape[1]

    def get_figsize(self, colorbar=True, nrows=1, ncols=1):
        cbar_width = self.cbar_width if colorbar else 0
        w_pix = self.store.hdf.attrs['naxis1']
        h_pix = self.store.hdf.attrs['naxis2']
        width = w_pix * self.inch_per_pix * ncols + cbar_width
        height = h_pix * self.inch_per_pix * nrows
        pad_l, pad_r, pad_b, pad_t = self.edge_pads
        width += pad_r + pad_r
        height += pad_b + pad_t
        return width, height

    def format_labels_for_grid(self, ax):
        if ax.is_first_col():
            self.set_lat_label(ax, labelpad=0.8)
        else:
            wax = ax.coords['dec']
            wax.set_axislabel('')
            wax.set_ticklabel_visible(False)
        if ax.is_last_row():
            self.set_lon_label(ax)
        else:
            wax = ax.coords['ra']
            wax.set_axislabel('')
            wax.set_ticklabel_visible(False)

    def subplots_adjust(self, **kwargs):
        if not kwargs:
            kwargs = self.adjust_kwargs
        plt.tight_layout()
        plt.subplots_adjust(**kwargs)

    def save(self, outname, dpi=300):
        save_figure(self.plot_dir/outname, dpi=dpi)

    def add_beam(self, ax, offset_frac=0.02, loc='upper left', color='cyan'):
        hdr = self.store.read_header()
        scale, _ = self.pixel_scale
        try:
            bmaj = hdr['BMAJ'] / scale
            bmin = hdr['BMIN'] / scale
            pa   = hdr['BPA']
            # convert axes frac to data position
        except KeyError:
            print('KeyError: undefined beam FITS header parameters in store')
            return
        xy = self.offset_corner_pos(offset_frac, loc=loc)
        ellipse = patches.Ellipse(xy, bmaj, bmin, pa, color=color)
        ax.add_artist(ellipse)

    def add_field_mask_contours(self, ax):
        nbest = self.store.hdf['products/nbest'][...]
        # The `nbest` array has values equal to -1 where the union of data cubes
        # contains NaNs.
        ax.contour(nbest, levels=[-0.5], colors='black', linewidths=0.7,
                linestyles='dotted')


##############################################################################
#                         Individual Plots
##############################################################################

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
    # save figure
    plt.tight_layout()
    save_figure(outname)


def plot_corner(group, outname='corner'):
    ncomp = group.attrs['ncomp']
    par_labels = TEX_LABELS.copy()
    par_labels[3] = r'$\log(N) \ [\log(\mathrm{cm^{-2}})]$'
    n_params = group.attrs['n_params'] // ncomp
    names = get_par_names()
    post = group['posteriors'][...][:,:-2]  # posterior param values
    # Give each model component parameter set its own sampler object so that
    # each can be over-plotted in its own color.
    samples = [
            getdist.MCSamples(
                samples=post[:,ii::ncomp],
                names=names,
                labels=par_labels,
                label=f'Component {ii+1}',
                name_tag=f'{ii}',
                sampler='nested')
            for ii in range(ncomp)
    ]
    [s.updateSettings({'contours': [0.68, 0.90]}) for s in samples]
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
    post = group['posteriors'][...][:,:-2]  # param values
    names = get_par_names(ncomp)
    par_labels = [''] * 12
    par_labels[0] = r'$v_\mathrm{lsr}\, (1) \ [\mathrm{km\,s^{-1}}]$'
    par_labels[1] = r'$v_\mathrm{lsr}\, (2) \ [\mathrm{km\,s^{-1}}]$'
    par_labels[8] = r'$\sigma_\mathrm{v}\, (1) \ [\mathrm{km\,s^{-1}}]$'
    par_labels[9] = r'$\sigma_\mathrm{v}\, (2) \ [\mathrm{km\,s^{-1}}]$'
    samples = getdist.MCSamples(samples=post, names=names, labels=par_labels,
            sampler='nested')
    samples.updateSettings({'contours': [0.68, 0.90]})
    fig = gd_plt.get_subplot_plotter()
    x_names = ['v1', 's1']
    y_names = ['v2', 's2']
    fig.rectangle_plot(x_names, y_names, roots=samples, filled=True,
            line_args={'lw': 2, 'color': 'tab:gray'})
    fig.export(f'{outname}.pdf')
    plt.close()


def plot_evdiff(sp, outname='evdiff', conv=True):
    if conv:
        prefix = 'conv'
        evid = sp.store.hdf['/products/conv_evidence'][...]
    else:
        prefix = 'local'
        evid = sp.store.hdf['/products/evidence'][...]
    data = evid[1] - evid[0]
    fig, ax = plt.subplots(figsize=sp.get_figsize(), subplot_kw={'projection': sp.wcs})
    im = ax.imshow(data, vmin=-3, vmax=3, cmap=CLR_CMAP)
    ax.contourf(data, levels=[3, 11, np.nanmax(data)],
            colors=['forestgreen', 'limegreen'])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\log \mathcal{Z}_1 / \mathcal{Z}_0$')
    sp.add_field_mask_contours(ax)
    sp.add_beam(ax)
    sp.set_labels(ax)
    sp.subplots_adjust()
    sp.save(f'{outname}_{prefix}')


def plot_nbest(sp, outname='nbest'):
    data = sp.store.hdf['/products/nbest']
    fig, ax = plt.subplots(figsize=sp.get_figsize(), subplot_kw={'projection': sp.wcs})
    im, cbar = sp.imshow_discrete(ax, data, vmin=0, vmax=2)
    cbar.set_label(r'$N_\mathrm{comp}$')
    sp.add_field_mask_contours(ax)
    sp.add_beam(ax)
    sp.set_labels(ax)
    sp.subplots_adjust()
    sp.save(outname)


def plot_conv_nbest(sp, outname='conv_nbest'):
    data = sp.store.hdf['/products/conv_nbest']
    fig, ax = plt.subplots(figsize=sp.get_figsize(), subplot_kw={'projection': sp.wcs})
    im, cbar = sp.imshow_discrete(ax, data, vmin=0, vmax=2)
    cbar.set_label(r'$N_\mathrm{comp}\ (\mathrm{conv.})$')
    sp.add_field_mask_contours(ax)
    sp.add_beam(ax)
    sp.set_labels(ax)
    sp.subplots_adjust()
    sp.save(outname)


def plot_deblend_peak(sp, outname='hf_deblend_peak'):
    labels = (r'(1,1)', r'(2,2)')
    data = sp.store.hdf['/products/peak_intensity']
    data = np.nanmax(data, axis=1)
    n_trans, _, _ = data.shape
    figsize = sp.get_figsize(nrows=1, ncols=2)
    fig, axes = plt.subplots(nrows=1, ncols=n_trans, figsize=figsize,
            subplot_kw={'projection': sp.wcs})
    vmin = 0
    vmax = np.nanmax(data)
    for ii, ax in enumerate(axes):
        img = data[ii,:,:]
        im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=HOT_CMAP)
        sp.set_corner_label(ax, labels[ii])
        sp.format_labels_for_grid(ax)
        sp.add_field_mask_contours(ax)
        sp.add_beam(ax)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.8])  # l, b, w, h
    cbar = plt.colorbar(im, cax=cax)
    cbar.minorticks_on()
    cbar.set_label(r'$\mathrm{max}(\tilde{T}_\mathrm{b}) \ [\mathrm{K}]$')
    sp.subplots_adjust(left=0.09, right=0.9, bottom=0.15, top=0.95)
    sp.save(outname)


def plot_deblend_intintens(sp, vmax=10, outname='hf_deblend_intintens'):
    """
    Figure preferences chosen to correspond with Figure 19 in Keown et al. (2019).

    Parameters
    ----------
    vmax : number, default 10
    """
    labels = (r'(1,1)', r'(2,2)')
    data = sp.store.hdf['/products/integrated_intensity']
    mask = sp.store.hdf['/products/peak_intensity'][:,0,:,:]  # first component
    data = np.nansum(data, axis=1)
    data[np.isnan(mask)] = np.nan
    n_trans, _, _ = data.shape
    figsize = sp.get_figsize(nrows=1, ncols=2)
    fig, axes = plt.subplots(nrows=1, ncols=n_trans, figsize=figsize,
            subplot_kw={'projection': sp.wcs})
    vmin = 0
    if vmax is None:
        vmax = np.nanmax(data)
        extend = 'neither'
    else:
        extend = 'max'
    for ii, ax in enumerate(axes):
        img = data[ii,:,:]
        im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=STR_CMAP)
        sp.set_corner_label(ax, labels[ii])
        sp.format_labels_for_grid(ax)
        sp.add_field_mask_contours(ax)
        sp.add_beam(ax)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.8])  # l, b, w, h
    cbar = plt.colorbar(im, cax=cax, extend=extend)
    cbar.minorticks_on()
    cbar.set_label(r'$\int \tilde{T}_\mathrm{b} \mathop{}\!\mathrm{d} v \ [\mathrm{K\, km\, s^{-1}}]$')
    sp.subplots_adjust(left=0.09, right=0.9, bottom=0.15, top=0.95)
    sp.save(outname)


def plot_ncomp_metrics(sp, outname='ncomp_metrics'):
    aic  = sp.store.hdf['/products/AIC'][...]
    aicc = sp.store.hdf['/products/AICc'][...]
    bic  = sp.store.hdf['/products/BIC'][...]
    lnz  = sp.store.hdf['/products/evidence'][...]
    figsize = sp.get_figsize(nrows=2, ncols=2)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize,
            subplot_kw={'projection': sp.wcs})
    all_labels = ('AIC', 'AICc', 'BIC', 'Evidence')
    all_data = (-aic, -aicc, -bic, lnz)
    #all_thresh = (35, 35, 35, 11)
    all_thresh = (5, 5, 5, 11)
    ncomp_max = sp.ncomp_max
    vmin, vmax = 0, ncomp_max
    for ax, data, thresh, label in zip(
            axes.flatten(), all_data, all_thresh, all_labels):
        nbest = np.full(data[0].shape, 0, dtype=np.int32)
        sp.set_corner_label(ax, label)
        for i in range(ncomp_max):
            nbest[
                    (nbest == i) &
                    (data[i+1] - data[i] > thresh)
            ] += 1
            im = ax.imshow(nbest, vmin=vmin, vmax=vmax, cmap=NBD_CMAP)
            sp.format_labels_for_grid(ax)
            sp.add_field_mask_contours(ax)
            sp.add_beam(ax)
    sp.subplots_adjust(left=0.15, right=0.95, bottom=0.10, top=0.95)
    sp.save(outname)


def plot_map_props(sp, outname='map_props'):
    # dimensions (m, p, b, l)
    data = sp.store.hdf['/products/nbest_MAP'][...]
    n_mod, n_params, _, _ = data.shape
    # create plots for each model parameter
    figsize = sp.get_figsize(nrows=1, ncols=n_mod)
    for ii in range(n_params):
        fig, axes = plt.subplots(nrows=1, ncols=n_mod, figsize=figsize,
                subplot_kw={'projection': sp.wcs})
        vmin = np.nanmin(data[:,ii,:,:])
        vmax = np.nanmax(data[:,ii,:,:])
        for jj, ax in enumerate(axes):
            img = data[jj,ii,:,:]
            im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=RNB_CMAP)
            sp.format_labels_for_grid(ax)
            sp.add_field_mask_contours(ax)
            sp.add_beam(ax)
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.8])  # l, b, w, h
        cbar = plt.colorbar(im, cax=cax)
        cbar.minorticks_on()
        cbar.set_label(TEX_LABELS[ii])
        sp.subplots_adjust(left=0.09, right=0.9, bottom=0.15, top=0.95)
        sp.save(f'{outname}_par{ii}')


def plot_quan_props(sp, quan_ix=4, outname='props', conv=True):
    if conv:
        prefix = 'cpar'
        # dimensions (b, l)
        nbest = sp.store.hdf['/products/conv_nbest'][...]
        # dimensions (r, m, p, M, b, l)
        data = sp.store.hdf['/products/conv_marginals'][...]
        data = take_by_components(data, nbest)  # -> (m, p, M, b, l)
        data = data[:,:,quan_ix,:,:]            # -> (m, p, b, l)
    else:
        prefix = 'par'
        # dimensions (m, p, M, b, l)
        data = sp.store.hdf['/products/nbest_marginals'][:,:,quan_ix,:,:]
    n_mod, n_params, _, _ = data.shape
    # create plots for each model parameter
    for ii in range(n_params):
        figsize = sp.get_figsize(nrows=1, ncols=n_mod)
        fig, axes = plt.subplots(nrows=1, ncols=n_mod, figsize=figsize,
                subplot_kw={'projection': sp.wcs})
        vmin = np.nanmin(data[:,ii,:,:])
        vmax = np.nanmax(data[:,ii,:,:])
        for jj, ax in enumerate(axes):
            img = data[jj,ii,:,:]
            im = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=RNB_CMAP)
            sp.format_labels_for_grid(ax)
            sp.add_field_mask_contours(ax)
            sp.add_beam(ax)
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.8])  # l, b, w, h
        cbar = plt.colorbar(im, cax=cax)
        cbar.minorticks_on()
        cbar.set_label(TEX_LABELS[ii])
        sp.subplots_adjust(left=0.09, right=0.9, bottom=0.15, top=0.95)
        sp.save(f'{outname}_quan{quan_ix}_{prefix}{ii}')


def plot_3d_volume(sp, outname='volume_field_contour'):
    from mayavi import mlab as mvi
    db_data = sp.store.hdf['/products/hf_deblended'][...]
    data = np.nansum(db_data, axis=1)[0,...]
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    obj = mvi.contour3d(data, colormap='inferno', transparent=True, vmin=vmin,
            vmax=vmax)
    filen = str(sp.plot_dir/f'{outname}.pdf')
    mvi.savefig(filen)
    mvi.clf()


def plot_amm_specfit(sp, stack, pix, n_model=1, outname='specfit', cold=False,
        kind='map', zoom=False):
    assert kind in ('map', 'bestfit')
    lon_pix, lat_pix = pix
    group = sp.store.hdf[f'/pix/{lon_pix}/{lat_pix}/{n_model}']
    params = group[f'{kind}_params'][...]
    print(params)
    obs_spec = stack.get_arrays(*pix)
    xarrs = get_amm_psk_xarrs(stack)
    syn_spec = [SyntheticSpectrum(x, params, cold=cold) for x in xarrs]
    fig = plt.figure(figsize=(4, 5))
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1 = plt.subplot2grid((3, 1), (2, 0))
    axes = (ax0, ax1)
    for data, xarr, model, ax in zip(obs_spec, xarrs, syn_spec, axes):
        varr = model.varr
        ax.fill_between(varr, data, np.zeros_like(data), color='yellow',
                edgecolor='none', alpha=0.5)
        ax.plot(varr, data, 'k-', linewidth=0.7, drawstyle='steps-mid')
        ax.plot(varr, model.components.T, '-', color='magenta', linewidth=1.0, alpha=0.5)
        ax.plot(varr, model.sum_spec, '-', color='red', linewidth=1.0,
                drawstyle='steps-mid')
        ax.set_xlim(varr.value.min(), varr.value.max())
    if zoom:
        axes[0].set_xlim(-7, 7)
        axes[1].set_xlim(-7, 7)
    ymin = obs_spec[0].min() * 1.1
    ymax = obs_spec[0].max() * 1.1
    axes[0].set_ylim(ymin, ymax)
    axes[1].set_ylim(ymin*0.4-0.5, ymax*0.4-0.5)
    axes[1].set_xlabel(r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$')
    axes[1].set_ylabel(r'$T_\mathrm{mb} \ [\mathrm{K}]$')
    axes[0].annotate(r'$\mathrm{NH_3}\, (1,1)$', (0.03, 0.91), xycoords='axes fraction')
    axes[1].annotate(r'$\mathrm{NH_3}\, (2,2)$', (0.03, 0.80), xycoords='axes fraction')
    plt.tight_layout()
    sp.save(f'{outname}_{lon_pix}_{lat_pix}_{n_model}')


def plot_amm_specfit_nsrun(sp, stack, pix, n_model=1, n_draw=50,
        outname='nsrun', interval=500):
    lon_pix, lat_pix = pix
    group = sp.store.hdf[f'/pix/{lon_pix}/{lat_pix}/{n_model}']
    post = group['posteriors'][...]
    n_samples = post.shape[0]
    assert n_draw < n_samples
    obs_spec = stack.get_arrays(*pix)
    xarrs = get_amm_psk_xarrs(stack)
    def get_spec_set(i_lo, i_hi):
        return [[
                SyntheticSpectrum(x, p)
                for p in post[i_lo:i_hi]
            ]
            for x in xarrs
        ]
    def get_reshaped_spec(models, length):
        return np.array([
                s.components for s in models
        ]).T.reshape(-1, length*n_model)
    syn_spec_sets = get_spec_set(0, n_draw)
    fig = plt.figure(figsize=(4, 5))
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1 = plt.subplot2grid((3, 1), (2, 0))
    axes = (ax0, ax1)
    trans_lines = []
    for data, xarr, models, ax in zip(obs_spec, xarrs, syn_spec_sets, axes):
        varr = models[0].varr
        ax.fill_between(varr, data, np.zeros_like(data), color='yellow',
                edgecolor='none', alpha=0.5)
        ax.plot(varr, data, 'k-', linewidth=0.7, drawstyle='steps-pre')
        mod_spec = get_reshaped_spec(models, n_draw)
        lines = ax.plot(varr, mod_spec, '-', color='red',
                linewidth=0.5, alpha=0.5)
        trans_lines.append(lines)
        ax.set_xlim(varr.value.min(), varr.value.max())
    ymin = obs_spec[0].min() * 1.1
    ymax = obs_spec[0].max() * 1.1
    axes[0].set_ylim(ymin, ymax)
    axes[1].set_ylim(ymin*0.4-0.5, ymax*0.4-0.5)
    axes[1].set_xlabel(r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$')
    axes[1].set_ylabel(r'$T_\mathrm{mb} \ [\mathrm{K}]$')
    axes[0].annotate(r'$\mathrm{NH_3}\, (1,1)$', (0.03, 0.91), xycoords='axes fraction')
    axes[1].annotate(r'$\mathrm{NH_3}\, (2,2)$', (0.03, 0.80), xycoords='axes fraction')
    plt.tight_layout()
    def init():
        for lines, data in zip(trans_lines, obs_spec):
            nans = np.full(data.shape, np.nan)
            for line in lines:
                line.set_ydata(nans)
        return lines
    def animate(n):
        i_lo = n * n_draw
        if n_draw > n_samples - i_lo:
            n_anim = n_samples % n_draw
        else:
            n_anim = n_draw
        i_hi = i_lo + n_anim
        syn_spec_sets = get_spec_set(i_lo, i_hi)
        for models, line_set in zip(syn_spec_sets, trans_lines):
            mod_spec = get_reshaped_spec(models, n_anim)
            for data, line in zip(mod_spec.T, line_set):
                line.set_ydata(data)
        lines_flat = list(itertools.chain.from_iterable(trans_lines))
        return lines_flat
    n_extra = int(n_samples % n_draw == 0)
    n_frames = post.shape[0] // n_draw + n_extra
    ani = animation.FuncAnimation(
            fig, animate, init_func=init, frames=n_frames,
            interval=interval, blit=True,
    )
    out_path = sp.plot_dir / f'{outname}_{lon_pix}_{lat_pix}_{n_model}.mp4'
    ani.save(str(out_path), writer='ffmpeg', dpi=300)


def test_amm_predict_precision():
    spectra = get_test_spectra()
    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(4, 3))
    for i, (syn, ax) in enumerate(zip(spectra, axes)):
        amms = syn.to_ammspec()
        params = syn.params
        amm_predict(amms, params)
        amm_spec = amms.get_spec()
        syn_spec = syn.sum_spec
        is_close = np.allclose(amm_spec, syn_spec, rtol=1e-8, atol=1e-5)
        print(f':: Close? {is_close}')
        #diff = np.log10(np.abs(amm_spec - syn_spec)) - np.log10(syn_spec)
        #diff[syn_spec < 1e-3] = np.nan
        diff = np.log10(np.abs(amm_spec - syn_spec))
        diff[diff < -12] = np.nan
        print(':: max log10(diff)   =', np.nanmax(diff))
        #import ipdb; ipdb.set_trace()
        ax.plot(syn.varr, diff, 'k-', drawstyle='steps-mid', linewidth=0.7)
        scaled = (
                amm_spec / amm_spec.max()
                * (np.nanmax(diff) - np.nanmin(diff)) + np.nanmin(diff)
        )
        ax.plot(syn.varr, scaled , 'r-', drawstyle='steps-mid', linewidth=0.7,
                alpha=0.5)
    ax.set_xlim(-30, 30)
    #ax.set_ylim(-15,  0)
    ax.set_xlabel(r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$')
    ax.set_ylabel(r'$\log_\mathrm{10}\left( |\Delta T_\mathrm{b}| \right) \ [\mathrm{K}]$')
    plt.tight_layout()
    plt.savefig(Path('plots/cython_test_compare_precision.pdf'))
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


