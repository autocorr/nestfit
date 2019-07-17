#!/usr/bin/env python3
"""
Gaussian mixture fitting with dynesty. This module was written within the main
`nestfit` module but moved here after basing the code off of (Py)MultiNest
rather than dynesty as done here.

Later trials showed that the "sorting" method for choosing the centroid
velocity done below is incorrect and causes problems in the fits. However,
adapting the working and correct code from the MultiNest implentation shows an
extremely strong performance penalty to using dynesty for this application
under similar parameters (static, 400 live points, multi-ellipsoid, uniformly
sampled).
"""

from pathlib import Path

import numpy as np
from matplotlib import ticker
from matplotlib import pyplot as plt
import multiprocessing_on_dill as multiprocessing

import dynesty
from dynesty import plotting as dyplot
from numba import jit


plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)
plt.rc('xtick', direction='out', top=True)
plt.rc('ytick', direction='out', right=True)

ROOT_DIR = Path('/lustre/aoc/users/bsvoboda/temp/nestfit')
DATA_DIR = ROOT_DIR / Path('data')
PLOT_DIR = ROOT_DIR / Path('plots')


@jit(nopython=True)
def gauss(xax, amp, cen, std):
    prefactor = amp / np.sqrt(2 * np.pi * std**2)
    return prefactor * np.exp(-(xax - cen)**2 / (2 * std**2))


class GaussianMixture:
    def __init__(self, xax, amp=None, cen=None, std=None, rms=0.03):
        self.xax = xax.reshape( 1, -1)
        self.nchan = self.xax.shape[1]
        self.amp = amp.reshape(-1,  1)
        self.cen = cen.reshape(-1,  1)
        self.std = std.reshape(-1,  1)
        self.rms = rms
        self.components = self.gauss()
        self.true_mix = self.components.sum(axis=0)

    def gauss(self):
        prefactor = self.amp / np.sqrt(2 * np.pi * self.std**2)
        return prefactor * np.exp(-(self.xax - self.cen)**2 / (2 * self.std**2))

    def sample_noise(self):
        return self.true_mix + np.random.normal(scale=self.rms, size=self.nchan)


def test_mixture():
    return GaussianMixture(
        np.linspace(-6, 6, 100),
        amp=np.array([0.3, 0.5, 0.2]),
        cen=np.array([-1, 0, 3]),
        std=np.array([1.5, 1.0, 0.5]),
        rms=0.03,
    )


def plot_spec():
    mix = test_mixture()
    fig, ax = plt.subplots(figsize=(4, 3))
    xarr = mix.xax.flatten()
    ax.hlines(0, mix.xax.min(), mix.xax.max(), color='0.5', linestyle='dashed')
    ax.plot(xarr, mix.components.T, drawstyle='steps')
    ax.plot(xarr, mix.sample_noise(), color='black', drawstyle='steps')
    ax.plot(xarr, mix.true_mix, color='red', drawstyle='steps')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    plt.tight_layout()
    plt.savefig(PLOT_DIR/Path('gauss_spec.pdf'))
    plt.close('all')


@jit(nopython=True)
def loglike(theta, xax, sample, rms, ncomp):
    amp = theta[0*ncomp:1*ncomp].reshape(-1, 1)
    cen = theta[1*ncomp:2*ncomp].reshape(-1, 1)
    std = theta[2*ncomp:3*ncomp].reshape(-1, 1)
    model = gauss(xax, amp, cen, std).sum(axis=0)
    difsqsum = ((sample - model)**2).sum()
    size = sample.shape[0]
    lnL = -0.5 * size * (np.log(2 * np.pi * rms**2) + 1 / rms**2 * difsqsum)
    if not np.isfinite(lnL):
        return -1e300
    else:
        return lnL


@jit(nopython=True)
def prior_transform(utheta, ncomp):
    uamp = utheta[0*ncomp:1*ncomp]
    ucen = utheta[1*ncomp:2*ncomp]
    ucen.sort()
    ustd = utheta[2*ncomp:3*ncomp]
    amp =  0.77 * uamp + 0.03  # (0.03, 0.8)
    cen = 10.0  * ucen - 5.0   # (-5, 5)
    std =  2.7  * ustd + 0.3   # (0.3, 3)
    return np.concatenate((amp, cen, std))


def run_dynesty():
    ncomp = 2
    mix = test_mixture()
    sample = mix.sample_noise()
    queue_size = 16
    #pool = multiprocessing.Pool(processes=queue_size)
    dsampler = dynesty.DynamicNestedSampler(
            loglike,
            prior_transform,
            ndim=3*ncomp,
            logl_args=(mix.xax, sample, mix.rms, ncomp),
            ptform_args=(ncomp,),
            bound='multi',  # cubes
            sample='unif',  # rslice
            #queue_size=queue_size,
            #pool=pool,
            #nlive=1000,
    )
    dsampler.run_nested()
    return dsampler


def plot_traceplot(dsampler):
    mix = test_mixture()
    truths = np.concatenate(
            (mix.amp.flatten(), mix.cen.flatten(), mix.std.flatten())
    )
    labels = ['a1', 'a2', 'a3', 'c1', 'c2', 'c3', 's1', 's2', 's3']
    fig, axes = dyplot.traceplot(dsampler.results, truths=truths,
            labels=labels, fig=plt.subplots(9, 2, figsize=(8, 11)))
    fig.tight_layout()
    plt.savefig(PLOT_DIR/Path('test_traceplot_3gauss.pdf'))
    plt.close('all')


def plot_corner(dsampler):
    mix = test_mixture()
    truths = np.concatenate(
            (mix.amp.flatten(), mix.cen.flatten(), mix.std.flatten())
    )
    labels = ['a1', 'a2', 'a3', 'c1', 'c2', 'c3', 's1', 's2', 's3']
    plt.rc('font', size=6, family='serif')
    plt.rc('xtick', direction='in', top=True)
    plt.rc('ytick', direction='in', right=True)
    fig, axes = dyplot.cornerplot(dsampler.results,
            show_titles=True, labels=labels, fig=plt.subplots(9, 9, figsize=(8, 8)))
    plt.savefig(PLOT_DIR/Path('test_corner_3gauss.pdf'))
    plt.close('all')


def plot_runplot(dsampler):
    fig, axes = dyplot.runplot(dsampler.results,
            fig=plt.subplots(4, 1, figsize=(8, 8)))
    plt.tight_layout()
    plt.savefig(PLOT_DIR/Path('test_runplot_3gauss.pdf'))
    plt.close('all')


def plot_compare_spec(dsampler):
    mix = test_mixture()
    xarr = mix.xax.flatten()
    pars = np.median(dsampler.results.samples, axis=0)
    comps = [gauss(xarr, pars[ii], pars[ii+3], pars[ii+6]) for ii in range(3)]
    comp_sum = np.sum(comps, axis=0)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(xarr, mix.sample_noise(), color='black', drawstyle='steps', alpha=0.5)
    ax.plot(xarr, mix.components.T, color='magenta', drawstyle='steps', alpha=0.5)
    ax.plot(xarr, mix.true_mix, color='red', drawstyle='steps')
    ax.plot(xarr, np.array(comps).T, color='cyan', drawstyle='steps', alpha=0.5)
    ax.plot(xarr, comp_sum, color='dodgerblue', drawstyle='steps')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    plt.tight_layout()
    plt.savefig(PLOT_DIR/Path('test_gauss_spec_recovered.pdf'))
    plt.close('all')


def plot_all_diagnostics(dsampler):
    for func in (plot_traceplot, plot_corner, plot_runplot, plot_compare_spec):
        print(f':: {func.__name__}')
        func(dsampler)


