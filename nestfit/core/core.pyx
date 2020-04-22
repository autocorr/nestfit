#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from nestfit.core.math cimport (NAN, c_sqrt, c_log)
from nestfit.core.cmultinest cimport run as c_run_multinest

from scipy import (integrate, interpolate)


# NOTE "DEF" constants will be in-lined at compile time
DEF FWHM = 2.3548200450309493


cdef class Distribution:
    # Extension type declaration in `.pxd`
    def __init__(self, xax, pdf):
        assert xax[1] > xax[0]
        assert xax.shape == pdf.shape
        self.dx   = xax[1] - xax[0]
        self.xax = xax
        self.pdf = pdf
        self.size = xax.shape[0]
        self.xmin = np.min(xax)
        self.xmax = np.max(xax)
        cdf = integrate.cumtrapz(pdf, xax, initial=0)
        cdf /= cdf.max()
        self.cdf = cdf
        # Hack to ensure that all values in the CDF are strictly ascending and
        # hold no duplicate values -- necessary for the interpolation
        eps_cdf = cdf + np.arange(self.size) * 1e-16
        eps_cdf /= eps_cdf.max()
        # Inverse interpolation of the CDF to compute the PPF
        inv_cdf = interpolate.UnivariateSpline(eps_cdf, xax, k=3, s=0)
        u = np.linspace(0, 1, self.size)
        self.du = u[1] - u[0]
        self.ppf = inv_cdf(u)

    cdef double ppf_interp(self, double u):
        """
        Normal PPF interpolation:
          x -> linearly sampled probability values over (0, 1)
          y -> parameter value at that quantile
            => y' = m * x' + b
        """
        cdef:
            long i_lo, i_hi
            double x_lo, y_lo, y_hi, slope
        i_lo = <long>((self.size - 1) * u)
        i_hi = i_lo + 1
        x_lo = i_lo * self.du
        y_lo = self.ppf[i_lo]
        y_hi = self.ppf[i_hi]
        slope = (y_hi - y_lo) / self.du
        return slope * (u - x_lo) + y_lo

    cdef double cdf_interp(self, double u):
        """
        Inverse interpolate the unit-cube probability `u` "y"-value of the
        CDF onto the parameter axis or "x"-value.
        This inverse CDF interpolation:
          x -> linearly sampled parameter value over (xmin, xmax)
          y -> cumulative probability at that parameter value
            => x' = 1 / m * (y' - b)
        """
        cdef:
            long i, i_lo, i_hi
            double x_lo, x_hi, y_lo, y_hi, slope
        # If `u == 0` then leading zeros in the CDF will result in the
        # bisection method (rounding down) will result in i -> 0, which will be
        # incorrect. So, set `u` to a small positive number that will be
        # neglible in the interpolation.
        if u <= self.cdf[0]:
            u = 1e-64
        # Bi-section method to find index of the nearest value in CDF
        i_lo = 0
        i_hi = self.size
        i = i_hi // 2
        while i != i_lo:
            if u > self.cdf[i]:
                i_lo = i  # in upper half
            else:
                i_hi = i  # in lower half
            # recenter pivot
            i = (i_hi + i_lo) // 2
        i_lo = i if i < self.size else self.size - 1
        i_hi = i_lo + 1
        # Walk the array to find the nearest value
        #i = 0
        #for i in range(1, self.size):
        #    if self.cdf[i] > u:
        #        break
        #i_hi = i
        #i_lo = i_hi - 1
        x_lo = self.xax[i_lo]
        y_lo = self.cdf[i_lo]
        y_hi = self.cdf[i_hi]
        slope = (y_hi - y_lo) / self.dx
        return 1 / slope * (u - y_lo) + x_lo

    cdef void cdf_over_interval(self, double x_lo, double x_hi):
        cdef:
            long i, i_lo, i_hi
            double csum = 0.0
        # If this occurs, it's almost certainly a bug, but guard against it for
        # safety.
        if x_lo > x_hi:
            x_lo, x_hi = x_hi, x_lo
        # Determine high and low indicies from the interval and guard against
        # out of bounds values.
        i_lo = <long>((x_lo - self.xmin) / self.dx)
        if i_lo >= self.size:
            i_lo = self.size - 1
        elif i_lo < 0:
            i_lo = 0
        i_hi = <long>((x_hi - self.xmin) / self.dx)
        # Guard for if x_hi - x_lo < dx and indices are the same.
        if i_hi == i_lo:
            i_hi = i_lo + 1
        if i_hi > self.size:
            i_hi = self.size
        elif i_hi < 0:
            i_hi = 1
        # clear cdf outside interval
        for i in range(0, i_lo):
            self.cdf[i] = 0.0
        for i in range(i_hi, self.size):
            self.cdf[i] = 1.0
        # recompute the CDF over the interval using the trapezoidal rule
        if i_hi - i_lo == 1:
            self.cdf[i_lo] = 1.0
        else:
            self.cdf[i_lo] = 0.0
            for i in range(i_lo+1, i_hi):
                csum += 0.5 * (self.pdf[i] + self.pdf[i-1])
                self.cdf[i] = csum
        # re-normalize the CDF
        for i in range(i_lo, i_hi):
            self.cdf[i] /= csum


# FIXME The prior framework could use some revision, as the inherited classes
# do not follow the Liskov substition principle well. However all that is
# currently necessary is that an instance correctly implements the `.interp`
# method.

cdef class Prior:
    # Extension type declaration in `.pxd`
    def __init__(self, dist, p_ix):
        """
        Interpolate the inverse cumulative prior function using an equally
        spaced sampling along the x-axis. Values are linearly interpolated
        between adjacent points.

        Parameters
        ----------
        data : array-like
            Inverse cumulative prior function (see "percent pointile function"
            `.ppf` scipy statistical distributions)
        p_ix : int
            Index of the parameter in the model described by the prior.
            For the ammonia model these are:
                vcen: 0, trot: 1, tex: 2, ncol: 3, sigm: 4
        """
        assert p_ix >= 0
        self.dist = dist
        self.p_ix = p_ix
        self.n_param = 1

    cdef void interp(self, double *utheta, long n):
        cdef:
            long i
            long ix = self.p_ix * n
        for i in range(n):
            utheta[ix+i] = self.dist.ppf_interp(utheta[ix+i])


cdef class DuplicatePrior(Prior):
    cdef:
        long p_ix_dup

    def __init__(self, dist, p_ix, p_ix_dup):
        assert p_ix >= 0
        assert p_ix_dup >= 0
        self.p_ix = p_ix
        self.p_ix_dup = p_ix_dup
        self.dist = dist
        self.n_param = 2

    cdef void interp(self, double *utheta, long n):
        cdef:
            long i
            long ix = self.p_ix * n
            long ix_dup = self.p_ix_dup * n
            double v
        for i in range(n):
            v = self.dist.ppf_interp(utheta[ix+i])
            utheta[ix+i] = v
            utheta[ix_dup+i] = v


cdef class ConstantPrior(Prior):
    cdef:
        double value

    def __init__(self, value, p_ix):
        self.value = value
        self.p_ix = p_ix
        self.n_param = 1

    cdef void interp(self, double *utheta, long n):
        cdef:
            long i
            long ix = self.p_ix * n
        for i in range(n):
            utheta[ix+i] = self.value


cdef class OrderedPrior(Prior):
    cdef void interp(self, double *utheta, long n):
        cdef:
            long i
            long ix = self.p_ix * n
            double u, umin
        # Values are sampled from the prior distribution, but a strict
        # ordering of the components is enforced from left-to-right by
        # making the offsets conditional on the last value:
        #     umin      umax(=1)
        #     |--x---------|
        #        |----x----|
        #             |--x-|
        umin = 0.0
        for i in range(n):
            u = umin + (1 - umin) * utheta[ix+i]
            umin = u
            utheta[ix+i] = self.dist.ppf_interp(u)


cdef class SpacedPrior(Prior):
    cdef:
        Prior prior_indep, prior_depen

    def __init__(self, prior_indep, prior_depen):
        """
        Parameters
        ----------
        prior_indep : Prior
            Independent prior. The first sample is drawn from this
            distribution.
        prior_depen : Prior
            Dependent prior. Further samples are drawn from this distribution.
        """
        self.prior_indep = prior_indep
        self.prior_depen = prior_depen
        self.p_ix = self.prior_indep.p_ix
        self.n_param = 1

    cdef void interp(self, double *utheta, long n):
        cdef:
            long i
            long ix = self.p_ix * n
            double v
        v = self.prior_indep.dist.ppf_interp(utheta[ix])
        utheta[ix] = v
        # Draw initial value from independent prior distribution, then
        # draw offsets from the running value from the dependent prior
        # distribution, updated in `v` for (n-1) samples.
        for i in range(1, n):
            v = v + self.prior_depen.dist.ppf_interp(utheta[ix+i])
            utheta[ix+i] = v


cdef class CenSepPrior(Prior):
    cdef:
        Prior prior_cen, prior_sep

    def __init__(self, prior_cen, prior_sep):
        self.prior_cen = prior_cen
        self.prior_sep = prior_sep
        self.p_ix = self.prior_cen.p_ix
        self.n_param = 1

    cdef void interp(self, double *utheta, long n):
        cdef:
            long i
            long ix = self.p_ix * n
            double vcen, vsep
        vcen = self.prior_cen.dist.ppf_interp(utheta[ix])
        if n == 1:
            utheta[ix] = vcen
        elif n == 2:
            vsep = self.prior_sep.dist.ppf_interp(utheta[ix+1])
            utheta[ix  ] = vcen - 0.5 * vsep
            utheta[ix+1] = vcen + 0.5 * vsep
        else:
            # FIXME need to parametrize higher order systems
            pass


cdef class ResolvedWidthPrior(Prior):
    cdef:
        double scale, sep_scale
        Prior vcen_prior, sigm_prior

    def __init__(self, vcen_prior, sigm_prior, scale=1.5):
        """
        Parameters
        ----------
        vcen_pdf : ParamPdf
            Velocity centroid parameter PDF
        sigm_prior : Prior
            Velocity dispersion prior
        sep_scale : number, default 1
            Multiplicative scaling factor of the Gaussian FWHM to provide the
            minimum separation between components.
        """
        self.vcen_prior = vcen_prior
        self.sigm_prior = sigm_prior
        self.scale = scale
        self.sep_scale = FWHM * scale
        self.n_param = 2

    cdef void interp(self, double *utheta, long n):
        cdef:
            long i, ix_v, ix_s
            double v_lo, v_hi, sep, sep_tot, overf_factor
            double[10] min_seps
            Distribution vcen_dist = self.vcen_prior.dist
        # FIXME will error if n > 10
        ix_v = self.vcen_prior.p_ix * n
        ix_s = self.sigm_prior.p_ix * n
        v_lo = vcen_dist.xmin
        v_hi = vcen_dist.xmax
        # compute widths
        self.sigm_prior.interp(utheta, n)
        if n == 1:
            utheta[ix_v] = vcen_dist.ppf_interp(utheta[ix_v])
            return
        # compute minimum separations between model components
        sep_tot = 0.0
        min_seps[0] = 0.0
        for i in range(1, n):
            sep = self.sep_scale * c_sqrt(utheta[ix_s+i] * utheta[ix_s+i-1])
            sep_tot += sep
            min_seps[i] = sep
        # shrink the separations if the separation sum is larger than the full
        # velocity interval
        if sep_tot > v_hi - v_lo:
            overf_factor = (v_hi - v_lo) / sep_tot
            sep_tot = 0.0
            for i in range(n):
                min_seps[i] *= overf_factor
                sep_tot += min_seps[i]
        # draw new centroid values based on the separations and the previously
        # drawn velocity (i.e., iteratively draw "to the right")
        v_hi -= sep_tot
        for i in range(n):
            sep = min_seps[i]  # first min_sep -> 0
            v_lo += sep
            v_hi += sep
            vcen_dist.cdf_over_interval(v_lo, v_hi)
            v_lo = vcen_dist.cdf_interp(utheta[ix_v+i])
            utheta[ix_v+i] = v_lo


cdef class PriorTransformer:
    # Extension type declaration in `.pxd`
    def __init__(self, priors):
        """
        Evaluate the prior transformation functions on the unit cube. The
        `.c_transform` method is passed to MultiNest and called on each
        likelihood evaluation.

        Parameters
        ----------
        priors : array_like(Prior)
            Array of `Prior` instances (or sub-class).
        """
        n_prior = priors.shape[0]
        assert n_prior >= 1
        n_param = 0
        for i in range(n_prior):
            n_param += priors[i].n_param
        self.priors = priors
        self.n_prior = n_prior
        self.n_param = n_param

    cdef void c_transform(self, double *utheta, long ncomp):
        """
        Parameters
        ----------
        utheta : double*
            Pointer to parameter unit cube.
        ncomp : int
            Number of components. `utheta` should have dimension [6*n] for
            ammonia.
        """
        # NOTE may do unsafe writes if `utheta` does not have the same
        # size as the number of components for `ncomp`.
        cdef:
            long i
            Prior prior
        for i in range(self.n_prior):
            prior = self.priors[i]
            prior.interp(utheta, ncomp)

    def transform(self, double[::1] utheta, long ncomp):
        if self.n_param * ncomp == utheta.shape[0]:
            self.c_transform(&utheta[0], ncomp)
        else:
            shape = utheta.shape[0]
            raise ValueError(f'Invalid shape for ncomp={ncomp}: {shape}')


cdef class Spectrum:
    # Extension type declaration in `.pxd`
    def __init__(self, xarr, data, noise, rest_freq=None, trans_id=None):
        """
        Parameters
        ----------
        xarr : array
            x-axis array in Hz. Note that the frequency axis must be in
            ascending order.
        data : array
            intensity values in K (brightness temperature).
        noise : number
            The baseline RMS noise level in K (brightness temperature).
        rest_freq : number
            The rest frequency in Hz
        """
        assert noise > 0
        nu_chan = xarr[1] - xarr[0]
        assert nu_chan > 0
        self.xarr = xarr
        self.data = data
        self.noise = noise
        size = xarr.shape[0]
        self.size = size
        self.rest_freq = 0 if rest_freq is None else rest_freq
        self.trans_id = -1 if trans_id is None else trans_id
        self.nu_chan = nu_chan
        self.nu_min = xarr[0]
        self.nu_max = xarr[self.size-1]
        self.pred = np.zeros_like(data)
        self.tarr = np.zeros_like(data)
        self.prefactor = -self.size / 2 * np.log(2 * np.pi * noise**2)
        # NOTE `pred` is zeros when this is calculated. It is simply the
        # computed likelihood of a "zero constant" model or "null model".
        self.null_lnZ = self.c_loglikelihood()

    cdef double c_loglikelihood(self):
        cdef:
            long i
            double dev
            double lnL = 0.0
        for i in range(self.size):
            dev = self.data[i] - self.pred[i]
            lnL += dev * dev
        return self.prefactor - lnL / (2 * self.noise**2)

    @property
    def sum_spec(self):
        return np.nansum(self.pred)

    @property
    def max_spec(self):
        return np.nanmax(self.pred)

    @property
    def loglikelihood(self):
        return self.c_loglikelihood()

    def get_spec(self):
        return np.array(self.pred)


cdef class Runner:
    # Extension type declaration in `.pxd`
    cdef void c_loglikelihood(self, double *utheta, double *lnL):
        pass

    def loglikelihood(self, double[::1] utheta):
        cdef double lnL
        self.c_loglikelihood(&utheta[0], &lnL)
        return lnL

    def get_spectra(self):
        return np.array(self.spectra)


cdef class Dumper:
    cdef:
        object group
        bint no_dump
        long n_calls, n_samples
        double[::1] quantiles
        list marginal_cols

    def __init__(self, group, no_dump=False):
        """
        Parameters
        ----------
        group : h5py.Group
            HDF5 group instance which to write the MultiNest output
        no_dump : bool, default False
            Do not write output data to group, used for debugging purposes.
        """
        self.group = group
        self.no_dump = no_dump
        self.n_calls = 0
        self.n_samples = -1
        self.quantiles = np.array([
            0.00, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.00,
            1.58655254e-1, 0.84134475,  # 1-sigma credible interval
            2.27501319e-2, 0.97724987,  # 2-sigma credible interval
            1.34989803e-3, 0.99865010,  # 3-sigma credible interval
        ])
        self.marginal_cols = [
            'min', 'p01', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99', 'max',
            '1s_lo', '1s_hi', '2s_lo', '2s_hi', '3s_lo', '3s_hi',
        ]

    def calc_marginals(self, posteriors):
        # The last two columns of the posterior array are -2*lnL and X*L/Z
        return np.quantile(posteriors[:,:-2], self.quantiles, axis=0)

    def flush(self):
        self.group.file.flush()

    def append_attributes(self, **kwargs):
        for name, value in kwargs.items():
            self.group.attrs[name] = value

    def append_datasets(self, **kwargs):
        for name, data in kwargs.items():
            self.group.create_dataset(name, data=data)


cdef class Context:
    cdef:
        Runner runner
        Dumper dumper

    def __init__(self, Runner runner, Dumper dumper):
        self.runner = runner
        self.dumper = dumper


cdef void mn_loglikelihood(double *utheta, int *ndim, int *n_params,
        double *lnL, void *context):
    (<Context> context).runner.c_loglikelihood(utheta, lnL)


cdef void mn_dump(int *n_samples, int *n_live, int *n_params,
            double **phys_live, double **posterior, double **param_constr,
            double *max_loglike, double *lnZ, double *ins_lnZ,
            double *lnZ_err, void *context):
    # NOTE all multi-dimensional arrays have Fortran stride
    cdef:
        double n, k, nullL, bic, aic, aicc, null_bic, null_aic, null_aicc
        Dumper dumper = (<Context> context).dumper
        Runner runner = (<Context> context).runner
    dumper.n_calls += 1
    # The last two iterations will have the same number of samples, so
    # only write out the values on the last iteration.
    if dumper.n_samples != n_samples[0]:
        dumper.n_samples = n_samples[0]
        return
    if dumper.no_dump:
        return
    # Final call, write out parameters to HDF5 file.
    runner.run_lnZ = lnZ[0]
    group = dumper.group
    # run and spectrum atttributes
    group.attrs['ncomp']      = runner.ncomp
    group.attrs['null_lnZ']   = runner.null_lnZ
    group.attrs['n_chan_tot'] = runner.n_chan_tot
    # nested sampling attributes:
    group.attrs['n_samples']      = n_samples[0]
    group.attrs['n_live']         = n_live[0]
    group.attrs['n_params']       = n_params[0]
    group.attrs['global_lnZ']     = lnZ[0]
    group.attrs['global_lnZ_err'] = lnZ_err[0]
    group.attrs['max_loglike']    = max_loglike[0]
    group.attrs['marg_cols']      = dumper.marginal_cols
    group.attrs['marg_quantiles'] = dumper.quantiles
    # information criteria
    n = <double>(runner.n_chan_tot)
    k = <double>(runner.n_params)
    nullL = runner.null_lnZ
    maxL = max_loglike[0]
    bic  = c_log(n) * k - 2 * maxL
    aic  = 2 * k - 2 * maxL
    aicc = aic + (2 * k**2 + 2 * k) / (n - k - 1)
    null_bic  = c_log(n) * k - 2 * nullL
    null_aic  = 2 * k - 2 * nullL
    null_aicc = null_aic + (2 * k**2 + 2 * k) / (n - k - 1)
    group.attrs['BIC']  = bic
    group.attrs['AIC']  = aic
    group.attrs['AICc'] = aicc
    group.attrs['null_BIC']  = null_bic
    group.attrs['null_AIC']  = null_aic
    group.attrs['null_AICc'] = null_aicc
    # posterior samples
    post_shape = (n_samples[0], n_params[0]+2)
    post_arr = mn_dptr_to_ndarray(posterior, post_shape)
    group.create_dataset('posteriors', data=post_arr)
    group.create_dataset('marginals', data=dumper.calc_marginals(post_arr))
    # posterior statistics
    pcon_shape = (1, 4*n_params[0])
    pcon_arr = mn_dptr_to_ndarray(param_constr, pcon_shape)
    pcon_arr = pcon_arr.reshape(4, n_params[0])
    group.create_dataset('bestfit_params', data=pcon_arr[2])
    group.create_dataset('map_params', data=pcon_arr[3])


cdef np.ndarray mn_dptr_to_ndarray(double **data, tuple shape):
    """
    Create a numpy array view of the memory pointed to by MultiNest.  The numpy
    C-API is used to construct an ndarray directly with attributes appropriate
    for the `posterior` and `param_constr` data (2D, Fortran-ordered,
    double/float64). Note that the memory mapped by these views will be freed
    when MultiNest returns.

    Information on the calling convention for `np.PyArray_New` were found here:
        https://gist.github.com/jdfr/688507524b6b4163e4c0
    See also:
        https://gist.github.com/GaelVaroquaux/1249305
    """
    cdef:
        np.ndarray array
        np.npy_intp dims[2]
    dims[0] = <np.npy_intp> shape[0]
    dims[1] = <np.npy_intp> shape[1]
    # Parameters select for Fortran-ordered contiguous data through:
    #   strides NULL
    #   data non-zero non-NULL
    #   (flags & NPY_ARRAY_F_CONTIGUOUS) non-zero non-NULL
    array = np.PyArray_New(
            np.ndarray,          # subtype; not quite sure why this works
            2,                   # nd; number of dimensions
            dims,                # dims; array size for each dimensions
            np.NPY_DOUBLE,       # type_num; data type for C double
            NULL,                # strides; array of stride lengths
            data[0],             # data; pointer to data
            0,                   # itemsize; ignored for type of fixed size
            np.NPY_ARRAY_FARRAY, # flags; bit mask for array flags
            <object>NULL,        # obj; used for ndarray subtypes
    )
    # NOTE a reference count may need to be incremented here
    return array


def run_multinest(
        Runner runner, Dumper dumper,
        IS=False, mmodal=True, ceff=False, nlive=400,
        tol=0.5, efr=0.3, nClsPar=None, maxModes=100, updInt=10, Ztol=-1e90,
        root='results', seed=-1, pWrap=None, fb=False, resume=False,
        initMPI=False, outfile=False, logZero=-1e100, maxiter=0):
    """
    Call the MultiNest `run` function. The `runner` and `dumper` classes wrap
    methods to perform the prior transformation, likelihood function call, and
    outfile creation.

    Parameters
    ----------
    runner : `Runner`
    dumper : `Dumper`
    IS : bool, default False
        Perform Importance Nested Sampling? If set to True, multi-modal
        (`mmodal`) sampling will be set to False within MultiNest.
    mmodal : bool, default True
        Perform mode separation?
    ceff : bool, default False
        Run in constant efficiency mode?
    nlive : int, default 400
        Number of live points.
    tol : float, default 0.5
        Evidence tolerance factor.
    efr : float, default 0.3
        Sampling efficiency.
    nClsPar : int, default None
        Number of parameters to perform the clustering over. If `None` then
        the clustering will be performed over all of the parameters. If a
        smaller number than the total is chosen, then the first such
        parameters will be used for the clustering.
    maxModes : int
        The maximum number of modes.
    updInt : int
    Ztol : float, default -1e90
    root : str, default 'results'
        The basename for the output files. This has an internal limit of 1000
        characters in MultiNest.
    seed : int, default -1
        Seed for the random number generator. The default value of -1 will
        set the seed from the system time.
    pWrap : iterable
    fb : bool, default False
    resume : bool, default False
    initMPI : bool, default False
    outfile : bool, default False
    logZero : float, default -1e100
    maxiter : int, default 0
    """
    assert runner.ndim > 0
    assert nlive > 0
    assert tol > 0
    assert 0 < efr <= 1
    assert maxModes > 0
    assert updInt > 0
    assert Ztol is not None and np.isfinite(Ztol)
    assert logZero is not None and np.isfinite(logZero)
    assert maxiter >= 0
    cdef:
        Context context = Context(runner, dumper)
        int[:] pWrap_a = np.zeros(runner.n_params, dtype='i')
        char root_c_arr[1000]
    root_c_arr[:len(root)] = <bytes> root
    if pWrap is not None:
        pWrap_a[:] = np.array(pWrap, dtype='i')
    if nClsPar is None:
        nClsPar = runner.n_params
    if nClsPar > runner.n_params:
        raise ValueError('Number of clustering parameters must be less than total.')
    c_run_multinest(
        IS,
        mmodal,
        ceff,
        nlive,
        tol,
        efr,
        runner.ndim,
        runner.n_params,
        nClsPar,
        maxModes,
        updInt,
        Ztol,
        root_c_arr,
        seed,
        &pWrap_a[0],
        fb,
        resume,
        outfile,
        initMPI,
        logZero,
        maxiter,
        &mn_loglikelihood,
        &mn_dump,
        <void *>context,
    )


##############################################################################
#                                 Tests
##############################################################################

def test_distribution():
    # The odd-numbered x-axis is centered at ix=100 and thus the median value
    # should yield an x-value near zero.
    x = np.linspace(-4, 4, 201)
    y = np.exp(-0.5 * x**2)
    dist = Distribution(x, y)
    eps = 1e-15
    assert abs(dist.ppf[100]) < eps
    assert abs(dist.ppf_interp(0.5)) < eps
    assert abs(dist.cdf_interp(0.5)) < eps


def test_resolved_width_prior():
    cdef:
        double[::1] utheta
    v_x = np.linspace(-4, 4, 200)
    v_y = np.exp(-0.5 * v_x**2)
    v_y /= v_y.sum()
    dist = Distribution(v_x, v_y)
    vcen_prior = Prior(dist, 0)
    sigm_prior = ConstantPrior(0.3, 1)
    rw_prior = ResolvedWidthPrior(vcen_prior, sigm_prior)
    n = 3  # N components
    for i in range(5):
        utheta = np.random.uniform(size=2*n)
        utheta[n:] = 0.3
        print(f':: {i+1} before')
        for u in utheta:
            print(f'{u:.6f}', end=' ')
        rw_prior.interp(&utheta[0], n)
        print(f'\n:: {i+1} after')
        for u in utheta:
            print(f'{u:.6f}', end=' ')
        print('\n----')


