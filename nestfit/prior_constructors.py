#!/usr/bin/env python3

import numpy as np
import scipy as sp

from nestfit.core.core import (
        Distribution,
        Prior,
        ConstantPrior,
        DuplicatePrior,
        OrderedPrior,
        SpacedPrior,
        CenSepPrior,
        ResolvedCenSepPrior,
        ResolvedPlacementPrior,
        PriorTransformer,
)


def get_irdc_priors(size=500, vsys=0.0):
    """
    Evaluate the prior distributions and construct a `PriorTransformer`
    instance for use with MultiNest. These distributions are constructed for
    the IRDCs observed in Svoboda et al. (in prep).

    Parameters
    ----------
    size : int
        Number of even, linearly spaced samples in the distribution
    vsys : float
        Systemic velocity to center prior distribution about
    """
    u = np.linspace(0, 1, size)
    # prior distribution x axes
    # 0 voff [-4.00,   4.0] km/s  (centered on vsys)
    #   vdep [    D, D+3.0] km/s  (with offset "D")
    # 1 trot [ 7.00,  30.0] K
    # 2 tex  [ 2.80,  12.0] K
    # 3 ntot [12.50,  16.5] log(cm^-2)
    # 4 sigm [    C, C+2.0] km/s  (with min sigma "C")
    x_voff =  8.00 * u -  4.00 + vsys
    x_vdep =  3.00 * u +  0.70
    x_trot = 23.00 * u +  7.00
    x_tex  =  9.26 * u +  2.80
    x_ntot =  4.00 * u + 12.50
    x_sigm =  2.00 * u +  0.067
    # prior PDFs values
    f_voff = sp.stats.beta( 5.0, 5.0).pdf(u)
    f_vdep = sp.stats.beta( 1.5, 3.5).pdf(u)
    f_trot = sp.stats.beta( 3.0, 6.7).pdf(u)
    f_tex  = sp.stats.beta( 1.0, 2.5).pdf(u)
    f_ntot = sp.stats.beta(10.0, 8.5).pdf(u)
    f_sigm = sp.stats.beta( 1.5, 5.0).pdf(u)
    # and distribution instances
    d_voff = Distribution(x_voff, f_voff)
    d_vdep = Distribution(x_vdep, f_vdep)
    d_trot = Distribution(x_trot, f_trot)
    d_tex  = Distribution(x_tex,  f_tex)
    d_ntot = Distribution(x_ntot, f_ntot)
    d_sigm = Distribution(x_sigm, f_sigm)
    # interpolation values, transformed to the intervals:
    priors = np.array([
            #OrderedPrior(d_voff, 0),
            #SpacedPrior(Prior(d_voff, 0), Prior(d_vdep, 0)),
            ResolvedPlacementPrior(
                Prior(d_voff, 0),
                Prior(d_sigm, 4),
                scale=1.5,
            ),
            Prior(d_trot, 1),
            Prior(d_tex,  2),
            Prior(d_ntot, 3),
            #Prior(d_sigm, 4),
            ConstantPrior(0, 5),
    ])
    return PriorTransformer(priors)


def get_synth_priors(size=500):
    """
    Evaluate the prior distributions and construct a `PriorTransformer`
    instance for use with MultiNest. These distributions are constructed for
    the tests of synthetic ammonia spectra, as used in Keown et al. (2019) in
    S6.1 pg 19-20.

    Parameters
    ----------
    size : int
        Number of even, linearly spaced samples in the distribution
    """
    u = np.linspace(0, 1, size)
    # prior distribution x axes
    # 0 voff [-3.900,  3.90] km/s  (center of two comps)
    #   vsep [ 0.130,  2.70] km/s    (sep between comps)
    # 1 tkin [ 7.900, 25.10] K
    # 2 tex  [ 7.900, 25.10] K    (fixed to tkin in LTE)
    # 3 ntot [12.950, 14.55] log(cm^-2)
    # 4 sigm [ 0.075,  2.10] km/s    (scaled log-normal)
    # 5 orth [ 0.000,  0.00]             (fixed to zero)
    #x_voff = 10.200 * u -  5.1
    x_voff =  7.800 * u -  3.90
    x_vsep =  2.570 * u +  0.13
    x_tkin = 17.200 * u +  7.90
    x_ntot =  1.600 * u + 12.95
    x_sigm =  2.025 * u +  0.075
    # prior PDFs values
    f_voff = np.ones_like(u) / size
    f_vsep = np.ones_like(u) / size
    f_tkin = np.ones_like(u) / size
    f_ntot = np.ones_like(u) / size
    f_sigm = sp.stats.lognorm(1.0, scale=0.136).pdf(u)
    # and distribution instances
    d_voff = Distribution(x_voff, f_voff)
    d_vsep = Distribution(x_vsep, f_vsep)
    d_tkin = Distribution(x_tkin, f_tkin)
    d_ntot = Distribution(x_ntot, f_ntot)
    d_sigm = Distribution(x_sigm, f_sigm)
    # interpolation values, transformed to the intervals:
    fwhm = 2 * np.sqrt(2 * np.log(2))
    priors = np.array([
            ## Using resolved width
            #ResolvedPlacementPrior(
            #    Prior(d_voff, 0),
            #    Prior(d_sigm, 4),
            #    scale=1/fwhm,
            #),
            #DuplicatePrior(d_tkin, 1, 2),
            #Prior(d_ntot, 3),
            #ConstantPrior(0, 5),
            ## Using center-separation prior
            ResolvedCenSepPrior(
                Prior(d_voff, 0),
                Prior(d_vsep, 0),
                Prior(d_sigm, 4),
                scale=1/fwhm,
            ),
            DuplicatePrior(d_tkin, 1, 2),
            Prior(d_ntot, 3),
            ConstantPrior(0, 5),
    ])
    return PriorTransformer(priors)


