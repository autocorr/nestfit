#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from nestfit.core.math cimport (c_sqrt, c_abs, c_floor,
        fast_expn, calcExpTableEntries, fillErfTable)
from nestfit.core.core cimport (Spectrum, Runner)


# Initialize interpolation table entries for `fast_expn`
calcExpTableEntries(3, 8)
fillErfTable()


# Speed of light
DEF CKMS = 299792.458      # km/s


# TODO
# do we need a specific gaussian spectrum? or just use base?
# functionality for fitting multiple lines at different frequencies with the
# same width and centroid velocity


cdef void c_gauss_predict(Spectrum s, double *params, int ndim) nogil:
    cdef:
        int i, j
        int nu_lo_ix, nu_hi_ix
        int ncomp = ndim // 3
        double voff, sigm, peak
        double nu, nu_width, nu_cen, nu_denom
        double nu_cutoff, nu_lo, nu_hi
    for i in range(s.size):
        s.pred[i] = 0.0
    for i in range(ncomp):
        voff = params[        i]
        sigm = params[  ncomp+i]
        peak = params[2*ncomp+i]
        nu_width = c_abs(sigm / CKMS * s.rest_freq)
        nu_cen   = s.rest_freq * (1 - voff / CKMS)
        nu_denom = 0.5 / (nu_width * nu_width)
        # Gaussians are approximated by only computing them within the range
        # of `exp(-12.5)` (3.7e-6) away from the HF line center center.
        nu_cutoff = c_sqrt(12.5 / nu_denom)
        nu_lo = (nu_cen - s.nu_min - nu_cutoff)
        nu_hi = (nu_cen - s.nu_min + nu_cutoff)
        # Get the lower and upper indices then check bounds
        nu_lo_ix = <int>c_floor(nu_lo/s.nu_chan)
        nu_hi_ix = <int>c_floor(nu_hi/s.nu_chan)
        if nu_hi_ix < 0 or nu_lo_ix > s.size-1:
            continue
        nu_lo_ix = 0 if nu_lo_ix < 0 else nu_lo_ix
        nu_hi_ix = s.size-1 if nu_hi_ix > s.size-1 else nu_hi_ix
        # Calculate the Gaussian line profile over the interval
        for j in range(nu_lo_ix, nu_hi_ix):
            nu = s.xarr[j] - nu_cen
            s.pred[j] += peak * fast_expn(nu * nu * nu_denom)


def gauss_predict(Spectrum s, double[::1] params):
    c_gauss_predict(s, &params[0], params.shape[0])


cdef class GaussianRunner(Runner):
    pass


