#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

cimport cython

import h5py

import numpy as np
cimport numpy as np
np.import_array()

from nestfit.cmultinest cimport run as c_run_multinest


cdef extern from 'math.h' nogil:
    const double M_PI
    double c_abs 'abs' (double)
    double c_exp 'exp' (double)
    double c_sqrt 'sqrt' (double)
    double c_floor 'floor' (double)


cdef extern from 'fastexp.h' nogil:
    double fast_expn 'FastExp' (const float)
    void calcExpTableEntries(const int, const int)
    void fillErfTable()
calcExpTableEntries(3, 8)
fillErfTable()


# NOTE These "DEF" constants will be in-lined with their value wherever they
# appear in the code at compilation time.

# Ammonia rotation constants [Splat ID: 01709]
# Poynter & Kakar (1975), ApJS, 29, 87; [from pyspeckit]
DEF BROT = 298117.06e6
DEF CROT = 186726.36e6
# Coudert & Roueff (2006), A&A, 449
#DEF BROT = 298192.92e6
#DEF CROT = 186695.86e6

# Ammonia inversion transition rest frequencies
DEF NU11 = 23.6944955e9    # Hz, pyspeckit Issue 91: Erik's custom freq
DEF NU22 = 23.722633335e9  # Hz

# Einstein A values
DEF EA11 = 1.712e-7        #64*!pi**4/(3*h*c**3)*nu11**3*mu0**2*(1/2.)
DEF EA22 = 2.291e-7        #64*!pi**4/(3*h*c**3)*nu22**3*mu0**2*(2/3.)

# Speed of light
DEF CKMS = 299792.458      # km/s
DEF CCMS = 29979245800.0   # cm/s

# Other physical constants in CGS
DEF H    = 6.62607004e-27  # erg s, Planck's constant
DEF KB   = 1.38064852e-16  # erg/K, Boltzmann's constant
DEF TCMB = 2.7315          # K, T(CMB) from pyspeckit

# Levels to calculate the partition function over
DEF NLEV  = 51
DEF NPARA = 34  # out of 51
DEF NORTH = 17  # out of 51

# Number of hyperfine levels for a transition
DEF NHF11 = 18
DEF NHF22 = 21


# Initialize statically allocated C arrays. Note that "for each" style
# iteration over a pointer type requires the syntex `for _ in P[:N]` to
# specify the length.
cdef:
    # J quantum numbers for para states
    int *JPARA = [
        1,  2,  4,  5,  7,  8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26,
        28, 29, 31, 32, 34, 35, 37, 38, 40, 41, 43, 44, 46, 47, 49, 50,
    ]
    # velocity offsets of the hyperfine lines in km/s
    double *VOFF11 = [
        19.8513, 19.3159, 7.88669, 7.46967, 7.35132, 0.460409, 0.322042,
        -0.0751680, -0.213003, 0.311034, 0.192266, -0.132382, -0.250923,
        -7.23349, -7.37280, -7.81526, -19.4117, -19.5500,
    ]
    double *VOFF22 = [
        26.5263, 26.0111, 25.9505, 16.3917, 16.3793, 15.8642, 0.562503,
        0.528408, 0.523745, 0.0132820, -0.00379100, -0.0132820, -0.501831,
        -0.531340, -0.589080, -15.8547, -16.3698, -16.3822, -25.9505,
        -26.0111, -26.5263,
    ]
    # optical depth weights of the hyperfine lines
    double *TAU_WTS11 = [
        # Original weights from pyspeckit
        #0.0740740, 0.1481480, 0.0925930, 0.1666670, 0.0185190, 0.0370370,
        #0.0185190, 0.0185190, 0.0925930, 0.0333330, 0.3000000, 0.4666670,
        #0.0333330, 0.0925930, 0.0185190, 0.1666670, 0.0740740, 0.1481480,
        # Normalized weights
        0.03703694, 0.07407389, 0.04629643, 0.08333337, 0.00925949,
        0.01851847, 0.00925949, 0.00925949, 0.04629643, 0.01666648,
        0.14999978, 0.23333315, 0.01666648, 0.04629643, 0.00925949,
        0.08333337, 0.03703694, 0.07407389,
    ]
    double *TAU_WTS22 = [
        # Original values from pyspeckit
        #0.0041860, 0.0376740, 0.0209300, 0.0372090, 0.0260470, 0.0018600,
        #0.0209300, 0.0116280, 0.0106310, 0.2674420, 0.4996680, 0.1465120,
        #0.0116280, 0.0106310, 0.0209300, 0.0018600, 0.0260470, 0.0372090,
        #0.0209300, 0.0376740, 0.0041860,
        # Normalized weights
        0.00333330, 0.02999971, 0.01666651, 0.02962943, 0.02074116,
        0.00148111, 0.01666651, 0.00925935, 0.00846544, 0.21296341,
        0.39788440, 0.11666714, 0.00925935, 0.00846544, 0.01666651,
        0.00148111, 0.02074116, 0.02962943, 0.01666651, 0.02999971,
        0.00333330,
    ]


cdef inline double square(double x) nogil:
    return x * x


cdef class AmmoniaSpectrum:
    cdef:
        int size
        double noise, prefactor, null_lnZ
        double nu_chan, nu_min, nu_max
        double[::1] xarr, data, pred, tarr

    def __init__(self, xarr, data, noise):
        """
        Parameters
        ----------
        xarr : array
            x-axis array in Hz.
        data : array
            intensity values in K (brightness temperature).
        noise : number
            The baseline RMS noise level in K (brightness temperature).
        """
        assert noise > 0
        nu_chan = xarr[1] - xarr[0]
        assert nu_chan > 0
        self.xarr = xarr
        self.data = data
        self.noise = noise
        size = xarr.shape[0]
        self.size = size
        self.nu_chan = nu_chan
        self.nu_min = xarr[0]
        self.nu_max = xarr[self.size-1]
        self.pred = np.empty_like(data)
        self.tarr = np.empty_like(data)
        self.prefactor = -self.size / 2 * np.log(2 * np.pi * noise**2)
        self.null_lnZ = self.loglikelihood()

    cdef double loglikelihood(self) nogil:
        cdef:
            int i
            double lnL = 0.0
        for i in range(self.size):
            lnL += square(self.data[i] - self.pred[i])
        return self.prefactor - lnL / (2 * square(self.noise))


cdef inline double partition_func(double trot) nogil:
    cdef:
        int j
        double Qtot = 0.0
    for j in JPARA[:NPARA]:
        Qtot += (
                (2 * j + 1)
                * fast_expn(H * (BROT * j * (j + 1)
                + (CROT - BROT) * j * j) / (KB * trot))
        )
    return Qtot


cdef void c_amm11_predict(AmmoniaSpectrum s, double *params, int ndim) nogil:
    cdef:
        int i, j, k
        int nu_lo_ix, nu_hi_ix
        int ncomp = ndim // 5
        double trot, tex, ntot, sigm, voff
        double Z11, Qtot, pop_rotstate, expterm, fracterm, widthterm, tau_main
        double hf_freq, hf_width, hf_offset, nu, T0, hf_tau_sum, tau_exp
        double nu_cutoff, nu_lo, nu_hi
        double hf_tau[NHF11]
        double hf_nucen[NHF11]
        double hf_idenom[NHF11]
    for i in range(s.size):
        s.pred[i] = 0.0
    for i in range(ncomp):
        voff = params[        i]
        trot = params[  ncomp+i]
        tex  = params[2*ncomp+i]
        ntot = params[3*ncomp+i]
        sigm = params[4*ncomp+i]
        # Calculate the partition function and the level populations
        Z11 = 3 * c_exp(-H * (BROT * 2 + (CROT - BROT)) / (KB * trot))
        Qtot = partition_func(trot)
        # Calculate the main line optical depth
        pop_rotstate = 10**ntot * Z11 / Qtot
        expterm = (
                (1 - c_exp(-H * NU11 / (KB * tex))) /
                (1 + c_exp(-H * NU11 / (KB * tex)))
        )
        fracterm = CCMS**2 * EA11 / (8 * M_PI * NU11**2)
        widthterm = (CKMS / (sigm * NU11 * c_sqrt(2 * M_PI)))
        tau_main = pop_rotstate * fracterm * expterm * widthterm
        # end of `ammonia.ammonia` --- start of `ammonia._ammonia_spectrum`
        # Calculate the velocity/frequency related constants for the
        # hyperfine transitions.
        for j in range(NHF11):
            hf_freq     = (1 - VOFF11[j] / CKMS) * NU11
            hf_width    = c_abs(sigm / CKMS * hf_freq)
            hf_offset   = voff / CKMS * hf_freq
            hf_nucen[j] = hf_freq - hf_offset
            hf_tau[j]   = tau_main * TAU_WTS11[j]
            hf_idenom[j] = 1 / (2.0 * square(hf_width))
        # For each HF line, sum the optical depth in each channel. The
        # Gaussians are approximated by only computing them within the range
        # of `exp(-20)` (4e-8) away from the HF line center center.
        for j in range(s.size):
            s.tarr[j] = 0
        for j in range(NHF11):
            nu_cutoff = c_sqrt(20 / hf_idenom[j])
            nu_lo = (hf_nucen[j] - s.nu_min - nu_cutoff)
            nu_hi = (hf_nucen[j] - s.nu_min + nu_cutoff)
            # Get the lower and upper indices then check bounds
            nu_lo_ix = <int>c_floor(nu_lo/s.nu_chan)
            nu_hi_ix = <int>c_floor(nu_hi/s.nu_chan)
            if nu_hi_ix < 0 or nu_lo_ix > s.size-1:
                continue
            nu_lo_ix = 0 if nu_lo_ix < 0 else nu_lo_ix
            nu_hi_ix = s.size-1 if nu_hi_ix > s.size-1 else nu_hi_ix
            # Calculate the Gaussian tau profile over the interval
            for k in range(nu_lo_ix, nu_hi_ix):
                nu = s.xarr[k] - hf_nucen[j]
                tau_exp = nu * nu * hf_idenom[j]
                s.tarr[k] += hf_tau[j] * fast_expn(tau_exp)
        # Compute the brightness temperature
        for j in range(s.size):
            nu = s.xarr[j]
            T0 = H * nu / KB
            s.pred[j] += (
                (T0 / (c_exp(T0 / tex) - 1) - T0 / (c_exp(T0 / TCMB) - 1))
                * (1 - fast_expn(s.tarr[j]))
            )


cdef void c_amm22_predict(AmmoniaSpectrum s, double *params, int ndim) nogil:
    cdef:
        int i, j, k
        int nu_lo_ix, nu_hi_ix
        int ncomp = ndim // 5
        double trot, tex, ntot, sigm, voff
        double Z22, Qtot, pop_rotstate, expterm, fracterm, widthterm, tau_main
        double hf_freq, hf_width, hf_offset, nu, T0, hf_tau_sum, tau_exp
        double nu_cutoff, nu_lo, nu_hi
        double hf_tau[NHF22]
        double hf_nucen[NHF22]
        double hf_idenom[NHF22]
    for i in range(s.size):
        s.pred[i] = 0.0
    for i in range(ncomp):
        voff = params[        i]
        trot = params[  ncomp+i]
        tex  = params[2*ncomp+i]
        ntot = params[3*ncomp+i]
        sigm = params[4*ncomp+i]
        # Calculate the partition function and the level populations
        Z22 = 5 * c_exp(-H * (BROT * 6 + (CROT - BROT) * 4) / (KB * trot))
        Qtot = partition_func(trot)
        # Calculate the main line optical depth
        pop_rotstate = 10**ntot * Z22 / Qtot
        expterm = (
                (1 - c_exp(-H * NU22 / (KB * tex))) /
                (1 + c_exp(-H * NU22 / (KB * tex)))
        )
        fracterm = CCMS**2 * EA22 / (8 * M_PI * NU22**2)
        widthterm = (CKMS / (sigm * NU22 * c_sqrt(2 * M_PI)))
        tau_main = pop_rotstate * fracterm * expterm * widthterm
        # end of `ammonia.ammonia` --- start of `ammonia._ammonia_spectrum`
        # Calculate the velocity/frequency related constants for the
        # hyperfine transitions.
        for j in range(NHF22):
            hf_freq     = (1 - VOFF22[j] / CKMS) * NU22
            hf_width    = c_abs(sigm / CKMS * hf_freq)
            hf_offset   = voff / CKMS * hf_freq
            hf_nucen[j] = hf_freq - hf_offset
            hf_tau[j]   = tau_main * TAU_WTS22[j]
            hf_idenom[j] = 1 / (2.0 * square(hf_width))
        # For each HF line, sum the optical depth in each channel. The
        # Gaussians are approximated by only computing them within the range
        # of `exp(-20)` (4e-8) away from the HF line center center.
        for j in range(s.size):
            s.tarr[j] = 0
        for j in range(NHF22):
            nu_cutoff = c_sqrt(20 / hf_idenom[j])
            nu_lo = (hf_nucen[j] - s.nu_min - nu_cutoff)
            nu_hi = (hf_nucen[j] - s.nu_min + nu_cutoff)
            # Get the lower and upper indices then check bounds
            nu_lo_ix = <int>c_floor(nu_lo/s.nu_chan)
            nu_hi_ix = <int>c_floor(nu_hi/s.nu_chan)
            if nu_hi_ix < 0 or nu_lo_ix > s.size-1:
                continue
            nu_lo_ix = 0 if nu_lo_ix < 0 else nu_lo_ix
            nu_hi_ix = s.size-1 if nu_hi_ix > s.size-1 else nu_hi_ix
            # Calculate the Gaussian tau profile over the interval
            for k in range(nu_lo_ix, nu_hi_ix):
                nu = s.xarr[k] - hf_nucen[j]
                tau_exp = nu * nu * hf_idenom[j]
                s.tarr[k] += hf_tau[j] * fast_expn(tau_exp)
        # Compute the brightness temperature
        for j in range(s.size):
            nu = s.xarr[j]
            T0 = H * nu / KB
            s.pred[j] += (
                (T0 / (c_exp(T0 / tex) - 1) - T0 / (c_exp(T0 / TCMB) - 1))
                * (1 - fast_expn(s.tarr[j]))
            )


def amm11_predict(AmmoniaSpectrum s, double[::1] params):
    c_amm11_predict(s, &params[0], params.shape[0])


def amm22_predict(AmmoniaSpectrum s, double[::1] params):
    c_amm22_predict(s, &params[0], params.shape[0])


cdef class Prior:
    cdef:
        int size
        double dx, dmin, dmax
        double[::1] data

    def __init__(self, data):
        """
        Interpolate the inverse cumulative prior function using an equally
        spaced sampling along the x-axis. Values are linearly interpolated
        between adjacent points.

        Parameters
        ----------
        data : array-like
            Inverse cumulative prior function (see "percent pointile function"
            `.ppf` scipy statistical distributions)
        """
        self.data = data
        self.size = data.shape[0]
        self.dx = 1 / <double>(self.size)
        self.dmin = np.min(data)
        self.dmax = np.max(data)

    cdef double _interp(self, double u) nogil:
        cdef:
            int i_lo, i_hi
            double x_lo, y_lo, y_hi, slope
        i_lo = <int>((self.size - 1) * u)
        i_hi = i_lo + 1
        x_lo = u - u % self.dx
        y_lo = self.data[i_lo]
        y_hi = self.data[i_hi]
        slope = (y_hi - y_lo) / self.dx
        return slope * (u - x_lo) + y_lo

    cdef void interp(self, double *utheta, int n, int npar) nogil:
        cdef int i
        for i in range(npar*n, (npar+1)*n):
            utheta[i] = self._interp(utheta[i])


cdef class OrderedPrior(Prior):
    cdef void interp(self, double *utheta, int n, int npar) nogil:
        cdef:
            int i
            double u, umin, umax
        # Values are sampled from the prior distribution, but a strict
        # ordering of the components is enforced from left-to-right by
        # making the offsets conditional on the last value:
        #     umin      umax
        #     |--x---------|
        #        |----x----|
        #             |--x-|
        umin, umax = 0.0, 1.0
        for i in range(npar*n, (npar+1)*n):
            u = umin = (umax - umin) * utheta[i] + umin
            utheta[i] = self._interp(u)


cdef class PriorTransformer:
    cdef:
        int npriors
        Prior p_trot, p_tex, p_ntot, p_sigm, p_voff

    def __init__(self, priors):
        """
        Evaluate the prior transformation functions on the unit cube. The
        `.transform` method is passed to MultiNest and called on each
        likelihood evaluation.

        Parameters
        ----------
        priors : iterable(Prior)
            List-like of individual `Prior` instances.
        """
        self.npriors = len(priors)
        self.p_voff = priors[0]
        self.p_trot = priors[1]
        self.p_tex  = priors[2]
        self.p_ntot = priors[3]
        self.p_sigm = priors[4]

    cdef void transform(self, double *utheta, int ncomp) nogil:
        """
        Parameters
        ----------
        utheta : double*
            Pointer to parameter unit cube.
        ncomp : int
            Number of components. `utheta` should have dimension [5*n].
        """
        # NOTE may do unsafe writes if `utheta` does not have the same
        # size as the number of components `n`.
        self.p_voff.interp(utheta, ncomp, 0)
        self.p_trot.interp(utheta, ncomp, 1)
        self.p_tex.interp( utheta, ncomp, 2)
        self.p_ntot.interp(utheta, ncomp, 3)
        self.p_sigm.interp(utheta, ncomp, 4)


cdef class Runner:
    cpdef double loglikelihood(self, object utheta, int ndim, int n_params):
        return 0.0

    cdef void c_loglikelihood(self, double *utheta, double *lnL):
        pass


cdef class AmmoniaRunner(Runner):
    cdef:
        PriorTransformer utrans
        AmmoniaSpectrum s11, s22
    cdef readonly:
        int ncomp, n_params, ndim, n_chan_tot
        double null_lnZ

    def __init__(self, spectra, utrans, ncomp=1):
        """
        Parameters
        ----------
        spectra : iterable(`AmmoniaSpectrum`)
            List of spectrum wrapper objects
        utrans : `PriorTransformer`
            Prior transformer class that samples the prior from the unit cube
            for the five model ammonia parameters.
        ncomp : int, default 1
            Number of velocity components

        Attributes
        ----------
        null_lnZ : number
            Natural log evidence for the "null model" of a constant equal to
            zero.
        """
        assert ncomp > 0
        self.s11 = spectra[0]
        self.s22 = spectra[1]
        self.utrans = utrans
        self.ncomp = ncomp
        self.n_params = 5 * ncomp
        self.ndim = self.n_params
        self.null_lnZ = self.s11.null_lnZ + self.s22.null_lnZ
        self.n_chan_tot = self.s11.size + self.s22.size

    cpdef double loglikelihood(self, object utheta, int ndim, int n_params):
        cdef:
            double lnL
            double[::1] params
        params = np.ctypeslib.as_array(utheta, shape=(n_params,))
        self.utrans.transform(&params[0], self.ncomp)
        c_amm11_predict(self.s11, &params[0], self.ndim)
        c_amm22_predict(self.s22, &params[0], self.ndim)
        lnL = self.s11.loglikelihood() + self.s22.loglikelihood()
        return lnL

    cdef void c_loglikelihood(self, double *utheta, double *lnL):
        self.utrans.transform(utheta, self.ncomp)
        c_amm11_predict(self.s11, utheta, self.ndim)
        c_amm22_predict(self.s22, utheta, self.ndim)
        lnL[0] = self.s11.loglikelihood() + self.s22.loglikelihood()


def check_hdf5_ext(store_name):
    if store_name.endswith('.hdf5'):
        return store_name
    else:
        return f'{store_name}.hdf5'


cdef class Dumper:
    cdef:
        str group_name, store_name
        bint no_dump
        int n_calls, n_samples
        double[::1] quantiles
        list marginal_cols

    def __init__(self, group_name, store_name='results', no_dump=False):
        self.group_name = group_name
        self.store_name = check_hdf5_ext(store_name)
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

    def append_attributes(self, **kwargs):
        with h5py.File(self.store_name, 'a') as hdf:
            group = hdf[self.group_name]
            for name, value in kwargs.items():
                group.attrs[name] = value

    def append_datasets(self, **kwargs):
        with h5py.File(self.store_name, 'a') as hdf:
            group = hdf[self.group_name]
            for name, data in kwargs.items():
                group.create_dataset(name, data=data)


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
    with h5py.File(dumper.store_name, 'a') as hdf:
        group = hdf.create_group(dumper.group_name)
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
        group.attrs['marginal_cols']  = dumper.marginal_cols
        # information criteria
        n = runner.n_chan_tot
        k = runner.n_params
        maxL = max_loglike[0]
        bic  = np.log(n) * k - 2 * maxL
        aic  = 2 * k - 2 * maxL
        aicc = aic + (2 * k**2 + 2 * k) / (n - k - 1)
        group.attrs['BIC']  = bic
        group.attrs['AIC']  = aic
        group.attrs['AICc'] = aicc
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
    C-API is used to construct a ndarray directly with attributes appropriate
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


