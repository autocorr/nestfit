#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

cimport cython
from libc.math cimport (M_PI, abs as c_abs, exp as c_exp, sqrt as c_sqrt)

import numpy as np
import scipy as sp
cimport numpy as np


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

# Planck's constant in CGS
DEF H    = 6.62607004e-27  # erg s
DEF KB   = 1.38064852e-16  # erg/K
DEF TCMB = 2.7315          # K, T(CMB) from pyspeckit

# Levels to calculate the partition function over
DEF NLEV  = 51
DEF NPARA = 34  # out of 51
DEF NORTH = 17  # out of 51

# Number of hyperfine levels for a transition
DEF NHF11 = 18
DEF NHF22 = 21


# Initialize stack allocated C arrays, note that these are raw data buffers and
# do not support broadcasting like numpy arrays or typed memory views.
cdef:
    # J quantum numbers for para states
    int JPARA[NPARA]
    # velocity offsets of the hyperfine lines in km/s
    double VOFF11[NHF11]
    double VOFF22[NHF22]
    # optical depth weights of the hyperfine lines
    double TAU_WTS11[NHF11]
    double TAU_WTS22[NHF22]
JPARA[:] = [
        1,  2,  4,  5,  7,  8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26,
        28, 29, 31, 32, 34, 35, 37, 38, 40, 41, 43, 44, 46, 47, 49, 50,
]
VOFF11[:] = [
        19.8513, 19.3159, 7.88669, 7.46967, 7.35132, 0.460409, 0.322042,
        -0.0751680, -0.213003, 0.311034, 0.192266, -0.132382, -0.250923,
        -7.23349, -7.37280, -7.81526, -19.4117, -19.5500,
]
VOFF22[:] = [
        26.5263, 26.0111, 25.9505, 16.3917, 16.3793, 15.8642, 0.562503,
        0.528408, 0.523745, 0.0132820, -0.00379100, -0.0132820, -0.501831,
        -0.531340, -0.589080, -15.8547, -16.3698, -16.3822, -25.9505,
        -26.0111, -26.5263,
]
TAU_WTS11[:] = [
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
TAU_WTS22[:] = [
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


cdef inline double square(double x):
    return x * x


cdef void c_amm11_predict(double[:] xarr, double[:] spec, double[:] params):
    cdef:
        int i, j, k
        int size = xarr.shape[0]
        int ncomp = params.shape[0] // 5
        double trot, tex, ntot, sigm, voff
        double Z11, Qtot, pop_rotstate, expterm, fracterm, widthterm, tau_main
        double hf_freq, hf_width, hf_offset, nu, T0, tau_hf_sum, tau_exp
        double hf_denom[NHF11]
        double hf_nucen[NHF11]
        double tau_hf[NHF11]
    spec[:] = 0
    for i in range(ncomp):
        voff = params[        i]
        trot = params[  ncomp+i]
        tex  = params[2*ncomp+i]
        ntot = params[3*ncomp+i]
        sigm = params[4*ncomp+i]
        # Calculate the partition function and the level populations
        Z11 = 3 * c_exp(-H * (BROT * 2 + (CROT - BROT)) / (KB * trot))
        Qtot = 0.0
        for j in range(NPARA):
            Qtot += (
                    (2 * JPARA[j] + 1)
                    * c_exp(-H * (BROT * JPARA[j] * (JPARA[j] + 1)
                    + (CROT - BROT) * JPARA[j]**2) / (KB * trot))
            )
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
            hf_denom[j] = 2.0 * hf_width**2
            hf_nucen[j] = hf_offset - hf_freq
            tau_hf[j]   = tau_main * TAU_WTS11[j]
        # For each channel in the spectrum compute the summed optical depth
        # over all of the hyperfine lines and then convert it to temperature
        # units.
        for j in range(size):
            nu = xarr[j]
            T0 = H * nu / KB
            tau_hf_sum = 0.0
            # Approximation to not include the contributions from Gaussian
            # components that are more than exp(-20) (4e-8) away from the HF
            # line center.
            for k in range(NHF11):
                tau_exp = (nu + hf_nucen[k])**2 / hf_denom[k]
                if tau_exp < 20:
                    tau_hf_sum += tau_hf[k] * c_exp(-tau_exp)
            spec[j] += (
                (T0 / (c_exp(T0 / tex) - 1) - T0 / (c_exp(T0 / TCMB) - 1))
                * (1 - c_exp(-tau_hf_sum))
            )


cdef void c_amm22_predict(double[:] xarr, double[:] spec, double[:] params):
    cdef:
        int i, j, k
        int size = xarr.shape[0]
        int ncomp = params.shape[0] // 5
        double trot, tex, ntot, sigm, voff
        double Z22, Qtot, pop_rotstate, expterm, fracterm, widthterm, tau_main
        double hf_freq, hf_width, hf_offset, nu, T0, tau_hf_sum, tau_exp
        double hf_denom[NHF22]
        double hf_nucen[NHF22]
        double tau_hf[NHF22]
    spec[:] = 0
    for i in range(ncomp):
        voff = params[        i]
        trot = params[  ncomp+i]
        tex  = params[2*ncomp+i]
        ntot = params[3*ncomp+i]
        sigm = params[4*ncomp+i]
        # Calculate the partition function and the level populations
        Z22 = 5 * c_exp(-H * (BROT * 6 + (CROT - BROT) * 4) / (KB * trot))
        Qtot = 0.0
        for j in range(NPARA):
            Qtot += (
                    (2 * JPARA[j] + 1)
                    * c_exp(-H * (BROT * JPARA[j] * (JPARA[j] + 1)
                    + (CROT - BROT) * JPARA[j]**2) / (KB * trot))
            )
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
            hf_denom[j] = 2.0 * hf_width**2
            hf_nucen[j] = hf_offset - hf_freq
            tau_hf[j]   = tau_main * TAU_WTS22[j]
        # For each channel in the spectrum compute the summed optical depth
        # over all of hte hyperfine lines and then convert it to temperature
        # units.
        for j in range(size):
            nu = xarr[j]
            T0 = H * nu / KB
            tau_hf_sum = 0.0
            # Approximation to not include the contributions from Gaussian
            # components that are more than exp(-20) (4e-8) away from the HF
            # line center.
            for k in range(NHF22):
                tau_exp = (nu + hf_nucen[k])**2 / hf_denom[k]
                if tau_exp < 20:
                    tau_hf_sum += tau_hf[k] * c_exp(-tau_exp)
            spec[j] += (
                (T0 / (c_exp(T0 / tex) - 1) - T0 / (c_exp(T0 / TCMB) - 1))
                * (1 - c_exp(-tau_hf_sum))
            )


cpdef void amm11_predict(double[:] xarr, double[:] spec, double[:] params):
    c_amm11_predict(xarr, spec, params)


cpdef void amm22_predict(double[:] xarr, double[:] spec, double[:] params):
    c_amm22_predict(xarr, spec, params)


cdef class PriorTransformer:
    cdef:
        int size
        double dx, vsys
        double[:] y_trot, y_tex, y_ntot, y_sigm, y_voff

    def __init__(self, int size=100, double vsys=0.0):
        """
        Evaluate the inverse cumulative prior functions and interpolate them
        using an equally spaced sampling along the x-axis. Values are linearly
        interpolated between adjacent points.

        Parameters
        ----------
        size : int
            Number of even, linearly spaced samples in the distribution
        vsys : double
            Systemic velocity to center prior distribution about
        """
        self.size = size
        self.dx = 1 / <double>(size)
        self.vsys = vsys
        # prior distributions
        x = np.linspace(0, 1-self.dx, size)
        dist_voff = sp.stats.beta(5.0, 5.0)
        dist_trot = sp.stats.gamma(4.4, scale=0.070)
        dist_tex  = sp.stats.beta(1.0, 2.5)
        dist_ntot = sp.stats.beta(16.0, 14.0)
        dist_sigm = sp.stats.gamma(1.5, loc=0.03, scale=0.2)
        # interpolation values, transformed to the intervals:
        # voff [-4.00,  4.0] km/s  (centered on vsys)
        # trot [ 7.00, 30.0] K
        # tex  [ 2.74, 12.0] K
        # ntot [12.00, 17.0] log(cm^-2)
        # sigm [ 0.00,  2.0] km/s
        self.y_voff =  8.00 * dist_voff.ppf(x) -  4.00 + vsys
        self.y_trot = 23.00 * dist_trot.ppf(x) +  7.00
        self.y_tex  =  9.26 * dist_tex.ppf(x)  +  2.74
        self.y_ntot =  5.00 * dist_ntot.ppf(x) + 12.00
        self.y_sigm =  2.00 * dist_sigm.ppf(x)

    cdef double _interp(self, double u, double[:] data):
        # FIXME may read out of bounds if `data` does not have same shape
        # as `self.size`.
        cdef:
            int i_lo, i_hi
            double x_lo, y_lo, y_hi, slope
        i_lo = <int>((self.size - 1) * u)
        i_hi = i_lo + 1
        x_lo = u - u % self.dx
        y_lo = data[i_lo]
        y_hi = data[i_hi]
        slope = (y_hi - y_lo) / self.dx
        return slope * (u - x_lo) + y_lo

    cpdef void transform(self, double[:] utheta, int n):
        # FIXME may do unsafe writes if `utheta` does not have the same size
        # as `n`.
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
        for i in range(  0,   n):
            u = umin = (umax - umin) * utheta[i] + umin
            utheta[i] = self._interp(u, self.y_voff)
        for i in range(  n, 2*n):
            utheta[i] = self._interp(utheta[i], self.y_trot)
        for i in range(2*n, 3*n):
            utheta[i] = self._interp(utheta[i], self.y_tex)
        for i in range(3*n, 4*n):
            utheta[i] = self._interp(utheta[i], self.y_ntot)
        for i in range(4*n, 5*n):
            utheta[i] = self._interp(utheta[i], self.y_sigm)


cdef class AmmoniaRunner:
    cdef:
        list spectra
        double[:] xarr11, xarr22, data11, data22, pred11, pred22
        PriorTransformer utrans
    cdef readonly:
        int ncomp, n_params
        double null_lnZ

    def __init__(self, spectra, utrans, ncomp=1):
        """
        Parameters
        ----------
        spectra : iterable
            List of spectrum wrapper objects
        utrans : `PriorTransformer`
            Prior transformer class that samples th prior from the unit cube
            for the five model ammonia parameters.
        ncomp : int, default 1
            Number of velocity components

        Attributes
        ----------
        null_lnZ : number
            Natural log evidence for the "null model" of a constant equal to
            zero.
        """
        self.spectra = spectra
        self.utrans = utrans
        self.ncomp = ncomp
        self.n_params = 5 * ncomp
        self.xarr11 = spectra[0].xarr.value.copy()
        self.xarr22 = spectra[1].xarr.value.copy()
        self.data11 = spectra[0].data.copy()
        self.data22 = spectra[1].data.copy()
        self.pred11 = np.empty_like(self.xarr11)
        self.pred22 = np.empty_like(self.xarr22)
        self.null_lnZ = np.sum([s.null_lnZ for s in self.spectra])

    cpdef double loglikelihood(self, object utheta, int ndim, int n_params):
        cdef:
            double lnL11, lnL22
            double[:] params
        params = np.ctypeslib.as_array(utheta, shape=(n_params,))
        self.utrans.transform(params, self.ncomp)
        c_amm11_predict(self.xarr11, self.pred11, params)
        c_amm22_predict(self.xarr22, self.pred22, params)
        lnL11, lnL22 = 0.0, 0.0
        for i in range(self.data11.shape[0]):
            lnL11 += square(self.data11[i] - self.pred11[i])
        for i in range(self.data22.shape[0]):
            lnL22 += square(self.data22[i] - self.pred22[i])
        lnL11 = self.spectra[0].loglikelihood(lnL11)
        lnL22 = self.spectra[1].loglikelihood(lnL22)
        return lnL11 + lnL22


