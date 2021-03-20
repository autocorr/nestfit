#cython: language_level=3

include "array_sizes.pxi"


cdef struct Transition:
    long n                    # Transition ID number (0 indexed)
    bint para                 # Is a para-transition?
    double nu                 # Rest frequency
    double ea                 # Einstein A coefficient
    long nhf                  # Number of hyperfine transitions
    double[MAX_HF_N] voff     # Velocity offsets of hyperfines
    double[MAX_HF_N] tau_wts  # Optical depth weights


cdef class Distribution:
    cdef:
        long size
        double du, dx, xmin, xmax
        double[::1] xax, pdf, cdf, ppf

    cdef double ppf_interp(self, double u)
    cdef double cdf_interp(self, double u)
    cdef void cdf_over_interval(self, double x_lo, double x_hi, double sfact)


cdef class Prior:
    cdef:
        long p_ix
        Distribution dist
    cdef readonly:
        long n_param

    cdef void interp(self, double *utheta, long n)


cdef class PriorTransformer:
    cdef:
        long n_prior
        Prior[:] priors
    cdef readonly:
        long n_param

    cdef void c_transform(self, double *utheta, long ncomp)


cdef class Spectrum:
    cdef:
        long size, trans_id
        double noise, prefactor, null_lnZ
        double rest_freq, nu_chan, nu_min, nu_max
        double[::1] xarr, data, pred, tarr

    cdef double c_loglikelihood(self)


cdef class HyperfineSpectrum(Spectrum):
    cdef:
        double[::1] tbg_arr
        Transition trans


cdef class Runner:
    cdef:
        PriorTransformer utrans
    cdef readonly:
        long n_model, ncomp, n_params, ndim, n_chan_tot
        double null_lnZ
    cdef public:
        double run_lnZ

    cdef void c_loglikelihood(self, double *utheta, double *lnL)


