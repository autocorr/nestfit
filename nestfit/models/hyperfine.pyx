#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

include "model_includes.pxi"

from nestfit.core.core cimport (Transition, HyperfineSpectrum)


DEF T0_SIZE = 1000
DEF T0_LO = H * 23.0e9 / KB  # Frequency in Hz
DEF T0_HI = H * 28.0e9 / KB
DEF T0_XMIN = T0_LO / 8.0    # Upper Tex = 8.0 K
DEF T0_XMAX = T0_HI / 2.7    # Lower Tex = 2.7 K
cdef:
    double[::1] T0_X = np.linspace(T0_XMIN, T0_XMAX, T0_SIZE)
    double[::1] T0_Y = 1.0 / (np.exp(T0_X) - 1.0)
    double T0_INV_DX = 1.0 / (T0_X[1] - T0_X[0])


cdef inline double c_iemtex_interp(double x) nogil:
    """
    Use a linear interpolation to approximate the function:
        f(x) = 1 / (exp(x) - 1)
    over the domain in `x` from (0.138, 0.498). This corresponds to frequency
    values between 23-28 GHz and excitation temperatures between 2.7-8.0 K.
    Outside of this interval, use the exact solution. Using a linear
    interpolation with N=1000 results in a relative numerical precision of
    ~1.8e-6 (|f'-f|/f) and a ~1.3x speed increase.
    """
    cdef:
        long i_lo, i_hi
        double x_lo, y_lo, y_hi, slope
    if T0_XMIN < x < T0_XMAX:
        i_lo = <long>((x - T0_XMIN) * T0_INV_DX)
        i_hi = i_lo + 1
        x_lo = T0_X[i_lo]
        y_lo = T0_Y[i_lo]
        y_hi = T0_Y[i_hi]
        slope = (y_hi - y_lo) * T0_INV_DX
        return slope * (x - x_lo) + y_lo
    else:
        return 1.0 / c_expm1(x)


def iemtex_interp(x):
    return c_iemtex_interp(x)


cdef void c_hf_predict(HyperfineSpectrum s, double voff, double tex, double ltau_main,
            double sigm) nogil:
    """See docstring for ``hf_predict``."""
    cdef:
        long i, j
        long nu_lo_ix, nu_hi_ix
        double tau_main
        double hf_freq, hf_width, hf_offset, nu, T0, hf_tau_sum, tau_exp
        double hf_tau, hf_nucen, hf_idenom
        double nu_cutoff, nu_lo, nu_hi
        Transition t = s.trans
    tau_main = 10.0**ltau_main
    # Calculate the velocity/frequency related constants for the hyperfine
    # transitions.
    for i in range(s.size):
        s.tarr[i] = 0.0
    for i in range(t.nhf):
        # Hyperfine (HF) dependent values
        hf_freq   = (1.0 - t.voff[i] / CKMS) * t.nu
        hf_width  = sigm / CKMS * hf_freq
        hf_offset = voff / CKMS * hf_freq
        hf_nucen  = hf_freq - hf_offset
        hf_tau    = tau_main * t.tau_wts[i]
        hf_idenom = 0.5 / (hf_width * hf_width)
        IF __APPROX:
            # For each HF line, sum the optical depth in each channel. The
            # Gaussians are approximated by only computing them within the
            # range of `exp(-12.5)` (5-sigma, 3.7e-6) away from the HF line
            # center.
            #   Eq:  exp(-nu**2 * hf_idenom) = exp(-12.5)
            nu_cutoff = c_sqrt(12.5 / hf_idenom)
            nu_lo = (hf_nucen - s.nu_min - nu_cutoff)
            nu_hi = (hf_nucen - s.nu_min + nu_cutoff)
            # Get the lower and upper indices then check bounds
            nu_lo_ix = <long>c_floor(nu_lo/s.nu_chan)
            nu_hi_ix = <long>c_floor(nu_hi/s.nu_chan)
            if nu_hi_ix < 0 or nu_lo_ix > s.size-1:
                continue
            nu_lo_ix = 0 if nu_lo_ix < 0 else nu_lo_ix
            nu_hi_ix = s.size-1 if nu_hi_ix > s.size-1 else nu_hi_ix
            # Calculate the Gaussian tau profile over the interval
            for j in range(nu_lo_ix, nu_hi_ix):
                nu = s.xarr[j] - hf_nucen
                tau_exp = nu * nu * hf_idenom
                s.tarr[j] += hf_tau * fast_expn(tau_exp)
        ELSE:
            for j in range(s.size):
                nu = s.xarr[j] - hf_nucen
                tau_exp = nu * nu * hf_idenom
                s.tarr[j] += hf_tau * c_exp(-tau_exp)
    # Compute the brightness temperature profile
    for i in range(s.size):
        if s.tarr[i] == 0.0:
            continue
        T0 = H * s.xarr[i] / KB
        # Eq: (T0 / (exp(T0 / tex) - 1) - T0 / (exp(T0 / TCMB) - 1))
        #     * (1 - exp(-tau))
        IF __APPROX:
            s.pred[i] += (
                T0 * (c_iemtex_interp(T0 / tex) - s.tbg_arr[i])
                * (1.0 - fast_expn(s.tarr[i]))
            )
        ELSE:
            s.pred[i] += (
                (T0 / c_expm1(T0 / tex) - T0 * s.tbg_arr[i])
                * (1.0 - c_exp(-s.tarr[i]))
            )


def hf_predict(HyperfineSpectrum s, voff, tex, ltau_main, sigm):
    """
    Predict the spectral profile for a line given the excitation temperature
    and main-line optical depth. The function results in a mutation of the
    ``Spectrum.pred`` array. The slabs are assumed to be optically thin with
    respect to each other.

    Parameters
    ----------
    s : Spectrum
    voff : number
        Line velocity offset
    tex : number
        Line excitation temperature
    ltau_main : number
        Log base-10 of the main line optical depth
    sigm : number
        Line velocity dispersion
    """
    c_hf_predict(s, voff, tex, ltau_main, sigm)


##############################################################################
#                                 Tests
##############################################################################

def test_iemtex_interp():
    fine_x = np.linspace(T0_XMIN, T0_XMAX, 100000)
    def f(x): return 1 / (np.exp(x) - 1)
    def diff(x): return np.abs((iemtex_interp(x) - f(x)) / f(x))
    diffs = np.array([diff(x) for x in fine_x])
    np.testing.assert_almost_equal(diffs.max(), 0, decimal=5)


