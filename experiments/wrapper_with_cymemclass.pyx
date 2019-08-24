#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

cimport cython
from cymem.cymem cimport Pool

import h5py
import scipy as sp

import numpy as np
cimport numpy as np
np.import_array()

from nestfit.cmultinest cimport run as c_run_multinest


cdef class AmmoniaSpectrum:
    cdef:
        Pool mem
        int size, trans, nhf
        double noise, prefactor, null_lnZ
        double nu_chan, nu_min, nu_max
        double *xarr
        double *data
        double *pred
        double *tarr
        double *hf_tau
        double *hf_nucen
        double *hf_idenom

    def __init__(self, object psk_spec, double noise, int trans=1):
        """
        Parameters
        ----------
        psk_spec : `pyspeckit.Spectrum`
        noise : number
            The baseline RMS noise level in K (brightness temperature).
        """
        cdef:
            int i
            double[::1] xarr, data
        assert noise > 0
        self.noise = noise
        self.trans = trans
        self.mem = Pool()
        xarr = psk_spec.xarr.as_unit('Hz').value.copy()
        data = psk_spec.data.data.copy()
        size = xarr.shape[0]
        self.size = size
        self.nu_chan = c_abs(xarr[1] - xarr[0])
        self.nu_min = xarr[0]
        self.nu_max = xarr[self.size-1]
        self.xarr = <double*>self.mem.alloc(size, sizeof(double))
        self.data = <double*>self.mem.alloc(size, sizeof(double))
        self.pred = <double*>self.mem.alloc(size, sizeof(double))
        self.tarr = <double*>self.mem.alloc(size, sizeof(double))
        for i in range(size):
            self.xarr[i] = xarr[i]
            self.data[i] = data[i]
            self.pred[i] = 0
            self.tarr[i] = 0
        if trans == 1:
            nhf = NHF11
        elif trans == 2:
            nhf = NHF22
        else:
            raise ValueError(f'Invalid inversion transition code: {trans}')
        self.nhf = nhf
        self.hf_tau = <double*>self.mem.alloc(nhf, sizeof(double))
        self.hf_nucen = <double*>self.mem.alloc(nhf, sizeof(double))
        self.hf_idenom = <double*>self.mem.alloc(nhf, sizeof(double))
        self.prefactor = -self.size / 2 * np.log(2 * np.pi * noise**2)
        self.null_lnZ = self.loglikelihood()

    cdef double loglikelihood(self) nogil:
        cdef:
            int i
            double lnL = 0.0
        for i in range(self.size):
            lnL += square(self.data[i] - self.pred[i])
        return self.prefactor - lnL / (2 * square(self.noise))


