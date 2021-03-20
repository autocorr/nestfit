#cython: language_level=3

from nestfit.core.core cimport HyperfineSpectrum


cdef void c_hf_predict(HyperfineSpectrum s, double voff, double tex, double ltau_main,
                double sigm) nogil


cdef double c_iemtex_interp(double x) nogil


