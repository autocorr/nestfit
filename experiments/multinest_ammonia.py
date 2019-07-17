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


class PriorTransformer:
    def __init__(self, size=100):
        x = np.linspace(0, 1, size)
        # prior distributions
        dist_trot = sp.stats.gamma(4.4, scale=0.070)
        dist_tex  = sp.stats.beta(1.0, 2.5)
        dist_ntot = sp.stats.beta(16.0, 14.0)
        dist_sigm = sp.stats.gamma(1.5, loc=0.03, scale=0.2)
        dist_voff = sp.stats.beta(5.0, 5.0)
        # interpolation functions, transformed to the intervals:
        # trot [ 7.00, 30.0] K
        # tex  [ 2.74, 12.0] K
        # ntot [12.00, 17.0] log(cm^-2)
        # sigm [ 0.00,  2.0] km/s
        # voff [-4.00,  4.0] km/s
        self.trot = self._interp(x, 23.00 * dist_trot.ppf(x) +  7.00)
        self.tex  = self._interp(x,  9.26 * dist_tex.ppf(x)  +  2.74)
        self.ntot = self._interp(x,  5.00 * dist_ntot.ppf(x) + 12.00)
        self.sigm = self._interp(x,  2.00 * dist_sigm.ppf(x)        )
        self.voff = self._interp(x,  8.00 * dist_voff.ppf(x) -  4.00)

    def _interp(self, x, y):
        return sp.interpolate.interp1d(x, y, kind='nearest',
                bounds_error=False, assume_sorted=True)


class AmmoniaRunner:
    model_name = 'ammonia'
    tex_labels = [
            r'$T_\mathrm{rot} \ [\mathrm{K}]$',
            r'$T_\mathrm{ex} \ [\mathrm{K}]$',
            r'$\log(N_\mathrm{tot}) \ [\mathrm{cm^{-2}}]$',
            r'$\sigma_\mathrm{v} \ [\mathrm{km\, s^{-1}}]$',
            r'$\Delta v \ [\mathrm{km\, s^{-1}}]$',
    ]

    def __init__(self, spectra, utrans, ncomp=1, vsys=0):
        """
        Parameters
        ----------
        spectra : iterable
            List of spectrum wrapper objects
        utrans : `PriorTransformer`
            Prior transformer object that exposes transforms for the five
            ammonia parameters.
        ncomp : int, default 1
            Number of velocity components
        vsys : number, default 0
            System velocity added to model offset

        Attributes
        ----------
        null_lnZ : number
            Natural log evidence for the "null model" of a constant equal to
            zero.
        """
        self.spectra = spectra
        self.utrans = utrans
        self.ncomp = ncomp
        self.vsys = vsys
        self.xarr11 = spectra[0].xarr.value.copy()
        self.xarr22 = spectra[1].xarr.value.copy()
        self.pred11 = np.empty_like(self.xarr11)
        self.pred22 = np.empty_like(self.xarr22)
        self.n_params = 5 * ncomp
        self.null_lnZ = np.sum([s.null_lnZ for s in self.spectra])

    @property
    def par_labels(self):
        comps = range(1, self.ncomp+1)
        return [
                f'{label}{n}'
                for label in ('Tk', 'Tx', 'N', 's', 'dv')
                for n in comps
        ]

    def loglikelihood(self, theta, ndim, n_params):
        n = self.ncomp
        params = np.ctypeslib.as_array(theta, shape=(n_params,))
        amm11_predict(self.xarr11, self.pred11, params)
        amm22_predict(self.xarr22, self.pred22, params)
        spec11, spec22 = self.spectra
        lnL11 = spec11.loglikelihood(
                np.sum((spec11.data - self.pred11)**2)
        )
        lnL22 = spec22.loglikelihood(
                np.sum((spec22.data - self.pred22)**2)
        )
        return lnL11 + lnL22

    def prior_transform(self, utheta, ndim, n_params):
        n = self.ncomp
        umin, umax = 0.0, 1.0
        for i in range(0, n):
            utheta[    i] = self.utrans.trot(utheta[    i])
            utheta[  n+i] = self.utrans.tex( utheta[  n+i])
            utheta[2*n+i] = self.utrans.ntot(utheta[2*n+i])
            utheta[3*n+i] = self.utrans.sigm(utheta[3*n+i])
            # Values are sampled from the prior distribution, but a strict
            # ordering of the components is enforced from left-to-right by
            # making the offsets conditional on the last value:
            #     umin      umax
            #     |--x---------|
            #        |----x----|
            #             |--x-|
            u = umin = (umax - umin) * utheta[4*n+i] + umin
            utheta[4*n+i] = self.utrans.voff(u)


