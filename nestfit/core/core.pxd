#cython: language_level=3


cdef class Distribution:
    cdef:
        int size
        double du, dx, xmin, xmax
        double[::1] xax, pdf, cdf, ppf

    cdef double ppf_interp(self, double u)
    cdef double cdf_interp(self, double u)
    cdef void cdf_over_interval(self, double x_lo, double x_hi)


cdef class Prior:
    cdef:
        int p_ix
        Distribution dist
    cdef readonly:
        int n_param

    cdef void interp(self, double *utheta, int n)


cdef class PriorTransformer:
    cdef:
        int n_prior
        Prior[:] priors
    cdef readonly:
        int n_param

    cdef void c_transform(self, double *utheta, int ncomp)


cdef class Spectrum:
    cdef:
        int size, trans_id
        double noise, prefactor, null_lnZ
        double rest_freq, nu_chan, nu_min, nu_max
        double[::1] xarr, data, pred, tarr

    cdef double c_loglikelihood(self)


cdef class Runner:
    cdef:
        PriorTransformer utrans
    cdef readonly:
        int n_model, ncomp, n_params, ndim, n_chan_tot
        double null_lnZ
    cdef public:
        double run_lnZ

    cdef void c_loglikelihood(self, double *utheta, double *lnL)


