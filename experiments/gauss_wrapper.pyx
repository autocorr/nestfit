#cython: language_level=3

cimport cython
from libc.math cimport exp
import ctypes

import numpy as np
cimport numpy as np


cdef class CGaussianModel:
    cdef readonly:
        int size, ncomp, n_params
        double noise, lnpin, null_lnZ
        double[:] xaxis, ydata

    def __init__(self, double[:] xaxis, double[:] ydata, double noise,
            int ncomp):
        self.xaxis = xaxis
        self.size  = xaxis.shape[0]
        self.ydata = ydata
        self.noise = noise
        self.ncomp = ncomp
        self.n_params = 3 * ncomp
        self.lnpin = -self.size / 2 * np.log(2 * np.pi * noise**2)
        self.null_lnZ = (
                self.lnpin - np.sum(np.ctypeslib.as_array(ydata)**2)
                / (2 * self.noise**2)
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef double loglikelihood(self, object cube_ptr, int ndim, int nparams):
        cdef:
            int n, i, j
            double y_dat, xval, y_mod, difsqsum, lnL
            double[:] amp, cen, std
            double[:] theta = np.ctypeslib.as_array(cube_ptr, shape=(nparams,))
        n = self.ncomp
        amp = theta[0  :  n]
        cen = theta[  n:2*n]
        std = theta[2*n:3*n]
        difsqsum = 0.0
        for i in range(self.size):
            y_dat = self.ydata[i]
            xval  = self.xaxis[i]
            y_mod = 0.0
            for j in range(n):
                y_mod += amp[j] * exp(-(xval - cen[j])**2 / (2 * std[j]**2))
            difsqsum += (y_mod - y_dat)**2
        lnL = self.lnpin - difsqsum / (2 * self.noise**2)
        return lnL

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef void prior_transform(self, object utheta, int ndim, int nparams):
        cdef:
            int i, n
            double v, vmin, vmax
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


