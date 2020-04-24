#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from nestfit.core.math cimport (M_PI, c_exp, c_expm1, c_sqrt, c_floor,
        c_log, fast_expn, calcExpTableEntries, fillErfTable)
from nestfit.core.core cimport (Spectrum, Runner)


# Initialize interpolation table entries for `fast_expn`
calcExpTableEntries(3, 8)
fillErfTable()

# Constants for conditional compilation.
# Use approximation methods versus exact terms. This applies to using:
#   * `fast_expn(x)` function in-place of `c_exp(-x)`
#   * calculation of the gaussian profiles over a limited neighborhood
#   * linear interpolation of the Tex-term in the brightness temperature axis
DEF __APPROX = True
# Use updated physical and spectroscopic constants.
DEF __NEW_CONST = True
# Execute code pathes for debugging
DEF __DEBUG = False

# Speed of light
DEF CKMS = 299792.458      # km/s
DEF CCMS = 29979245800.0   # cm/s

# Other physical constants in CGS; from `astropy.constants`
DEF H    = 6.62607015e-27  # erg s, Planck's constant
DEF KB   = 1.380649e-16    # erg/K, Boltzmann's constant
IF __NEW_CONST:
    DEF TCMB = 2.72548     # K, T(CMB); Fixsen (2009) ApJ 707 916F
ELSE:
    DEF TCMB = 2.7315      # K, T(CMB); from pyspeckit

# Ammonia rotation constants [Splat ID: 01709]
IF __NEW_CONST:
    # Coudert & Roueff (2006), A&A 449 855-859
    DEF BROT = 298192.92e6
    DEF CROT = 186695.86e6
ELSE:
    # Poynter & Kakar (1975), ApJS, 29, 87
    DEF BROT = 298117.06e6
    DEF CROT = 186726.36e6

# Levels to calculate the partition function over
DEF NPART = 51
DEF NPARA = 34  # out of 51
DEF NORTH = 17  # out of 51
# Maximum number of hyperfine transitions for fixed length arrays
DEF MAX_HF_N = 50
# Number of rotational levels: (1,1) thru (9,9)
DEF N_LEVELS = 9


# Initialize statically allocated C arrays with data values for each
# transition. Values taken from:
#   `pyspeckit.pyspeckit.spectrum.models.ammonia`
cdef:
    long i
    long JORTH[NORTH]
    long JPARA[NPARA]
    long NHF[N_LEVELS]
    double NU[N_LEVELS]
    double EA[N_LEVELS]
    double VOFF[N_LEVELS][MAX_HF_N]
    double TAU_WTS[N_LEVELS][MAX_HF_N]

# J quantum numbers for para states
JORTH = [i for i in range(NPART) if i % 3 == 0]
# J quantum numbers for ortho states
JPARA = [i for i in range(NPART) if i % 3 != 0]
# Number of hyperfine transitions per level
NHF = [
        18,  # (1,1)
        21,  # (2,2)
        26,  # (3,3)
        7,   # (4,4)
        7,   # (5,5)
        7,   # (6,6)
        7,   # (7,7)
        7,   # (8,8)
        1,   # (9,9)
]

# Ammonia inversion transition rest frequencies
# In Hz, note that for (1,1) Erik's custom freq is used,
# see pyspeckit issue 91.
NU = [
        23.6944955e9,    # (1,1)
        23.722633335e9,  # (2,2)
        23.8701296e9,    # (3,3)
        24.1394169e9,    # (4,4)
        24.53299e9,      # (5,5)
        25.05603e9,      # (6,6)
        25.71518e9,      # (7,7)
        26.51898e9,      # (8,8)
        27.477943e9,     # (9,9)
]

# Transition Einstein A values
IF __NEW_CONST:
    # Einstein A values are calculated using the expression:
    #   A = 64 * pi**4 / (3*h * c**3) * nu**3 * mu0**2 * (j / (j + 1))
    # The values computed below use the dipole moment found in
    # Coudert & Roueff (2006), A&A 449 855-859
    #   mu0 = 1.471 D     (note 1 Debye is 1e-18 statC cm)
    # in addition to up-to-date values for `h` and `c`.  These are
    # consistent with the JPL values on Splatalogue to approximately
    # 4 significant digits.
    EA = [
            1.67524303e-07,  # (1,1)
            2.24162441e-07,  # (2,2)
            2.56915917e-07,  # (3,3)
            2.83423417e-07,  # (4,4)
            3.09910019e-07,  # (5,5)
            3.39590403e-07,  # (6,6)
            3.74750461e-07,  # (7,7)
            4.17525824e-07,  # (8,8)
            4.70284410e-07,  # (9,9)
    ]
ELSE:
    # Values from pyspeckit, originally computed with:
    #   mu0 = 1.476 D  (Poynter & Kakar 1975; pg. 9)
    EA = [
            1.712e-7,        # (1,1)
            2.291e-7,        # (2,2)
            2.625e-7,        # (3,3)
            3.167e-7,        # (4,4)
            3.099109e-07,    # (5,5)
            3.395797e-07,    # (6,6)
            3.747560e-07,    # (7,7)
            4.175308e-07,    # (8,8)
            2.602045e-07,    # (9,9)
    ]

# Velocity offsets of the hyperfine transitions in km/s
VOFF[0][:NHF[0]] = [  # (1,1)
        19.851300,  19.315900,  7.8866900,  7.4696700,  7.3513200,
        0.4604090,  0.3220420, -0.0751680, -0.2130030,  0.3110340,
        0.1922660, -0.1323820, -0.2509230, -7.2334900, -7.3728000,
        -7.815260, -19.411700, -19.550000,
]
VOFF[1][:NHF[1]] = [  # (2,2)
        26.526300,  26.011100,  25.950500,  16.391700,  16.379300,
        15.864200,  0.5625030,  0.5284080,  0.5237450,  0.0132820,
       -0.0037910, -0.0132820, -0.5018310, -0.5313400, -0.5890800,
       -15.854700, -16.369800, -16.382200, -25.950500, -26.011100,
       -26.526300,
]
VOFF[2][:NHF[2]] = [  # (3,3)
        29.195098,  29.044147,  28.941877,  28.911408,  21.234827,
        21.214619,  21.136387,  21.087456,  1.0051220,  0.8060820,
        0.7780620,  0.6285690,  0.0167540, -0.0055890, -0.0134010,
       -0.6397340, -0.7445540, -1.0319240, -21.125222, -21.203441,
       -21.223649, -21.076291, -28.908067, -28.938523, -29.040794,
       -29.191744,
]
VOFF[3][:NHF[3]] = [  # (4,4)
        0.0, -30.49783692, 30.49783692, 0.0,  24.25907811,
       -24.25907811, 0.0,
]
VOFF[4][:NHF[4]] = [  # (5,5)
        31.4053287863, 26.0285409785, 0.0, 0.0, 0.0, -25.9063412556,
       -31.2831290633,
]
VOFF[5][:NHF[5]] = [  # (6,6)
        31.5872901302, 27.0406347326, 0.0, 0.0, 0.0, -26.9209859064,
       -31.4676413039,
]
VOFF[6][:NHF[6]] = [  # (7,7)
        31.3605314845, 27.3967468359, 0.0, 0.0, 0.0, -27.5133287373,
       -31.477113386,
]
VOFF[7][:NHF[7]] = [  # (8,8)
        30.9752235915, 27.4707274918, 0.0, 0.0, 0.0, -27.5837757531,
       -30.9752235915,
]
VOFF[8][:NHF[8]] = [  # (9,9)
        0.0,
]

# Optical depth weights of the hyperfine transitions, unitless.
# Weights are taken from pyspeckit and normalized.
# Note that the magnetic hyperfines are not included past (3,3).
# Note the precision on (1,1)-(3,3) is excessive, as the weights are only known
# to about 5-6 digits, but present for the purposes of testing the numerical
# accuracy against pyspeckit.
TAU_WTS[0][:NHF[0]] = [  # (1,1)
        3.7036944444583331e-02, 7.4073888889166661e-02,
        4.6296430555354165e-02, 8.3333374999937510e-02,
        9.2594861107708343e-03, 1.8518472222291665e-02,
        9.2594861107708343e-03, 9.2594861107708343e-03,
        4.6296430555354165e-02, 1.6666475000287499e-02,
        1.4999977500033751e-01, 2.3333315000027499e-01,
        1.6666475000287499e-02, 4.6296430555354165e-02,
        9.2594861107708343e-03, 8.3333374999937510e-02,
        3.7036944444583331e-02, 7.4073888889166661e-02,
]
TAU_WTS[1][:NHF[1]] = [  # (2,2)
        3.3333014814319341e-03, 2.9999713332887409e-02,
        1.6666507407159671e-02, 2.9629434979121079e-02,
        2.0741161893659245e-02, 1.4811134150653125e-03,
        1.6666507407159671e-02, 9.2593477367631464e-03,
        8.4654390943867397e-03, 2.1296340535048242e-01,
        3.9788439670906156e-01, 1.1666714444518766e-01,
        9.2593477367631464e-03, 8.4654390943867397e-03,
        1.6666507407159671e-02, 1.4811134150653125e-03,
        2.0741161893659245e-02, 2.9629434979121079e-02,
        1.6666507407159671e-02, 2.9999713332887409e-02,
        3.3333014814319341e-03,
]
TAU_WTS[2][:NHF[2]] = [  # (3,3)
        1.0733009496302131e-02, 7.3598529604831297e-03,
        3.0055577436436044e-03, 4.8085422957419802e-03,
        5.8220646798827188e-03, 7.7475821627062281e-03,
        4.3472933350838039e-03, 1.0143100958382566e-02,
        1.6829022799877465e-02, 9.0910682245853580e-03,
        9.4700450746138028e-03, 8.2989803509693240e-03,
        2.5670824033959128e-01, 4.0182836637346286e-01,
        1.5524222134698701e-01, 8.2989803509693240e-03,
        9.4700450746138028e-03, 1.6829022799877465e-02,
        4.3472933350838039e-03, 7.7475821627062281e-03,
        5.8220646798827188e-03, 1.0143100958382566e-02,
        4.8085422957419802e-03, 3.0055577436436044e-03,
        7.3598529604831297e-03, 1.0733009496302131e-02,
]
TAU_WTS[3][:NHF[3]] = [  # (4,4)
        0.2431, 0.0162, 0.0162, 0.3008, 0.0163, 0.0163, 0.3911,
]
TAU_WTS[4][:NHF[4]] = [  # (5,5)
        0.0109080940831, 0.0109433143618, 0.311493418617, 0.261847767275,
        0.382955997218,  0.0109433143618, 0.0109080940831,
]
TAU_WTS[5][:NHF[5]] = [  # (6,6)
        0.0078350431801, 0.00784948916416, 0.317644539734, 0.274246689798,
        0.376739705779, 0.00784948916416, 0.0078350431801,
]
TAU_WTS[6][:NHF[6]] = [  # (7,7)
        0.00589524944656, 0.00590204051181, 0.371879455317, 0.321515700951,
        0.283010263815, 0.00590204051181, 0.00589524944656,
]
TAU_WTS[7][:NHF[7]] = [  # (8,8)
        0.00459516014524, 0.00459939439378, 0.324116135075, 0.289534720829,
        0.367960035019, 0.00459939439378, 0.00459516014524,
]
TAU_WTS[8][:NHF[8]] = [  # (9,9)
        1.0,
]


cdef struct Transition:
    long n
    bint para
    double nu
    double ea
    long nhf
    double[MAX_HF_N] voff
    double[MAX_HF_N] tau_wts


# Fill global transition struct array
cdef:
    Transition[N_LEVELS] TRANS
for i in range(N_LEVELS):
    TRANS[i].n    = i + 1
    TRANS[i].para = (i + 1) % 3 != 0
    TRANS[i].nu   = NU[i]
    TRANS[i].ea   = EA[i]
    TRANS[i].nhf  = NHF[i]
    TRANS[i].voff = VOFF[i]
    TRANS[i].tau_wts = TAU_WTS[i]


PAR_NAMES = [
        'voff', 'trot', 'tex', 'ntot', 'sigm', 'orth',
]

TEX_LABELS = [
        r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$',
        r'$T_\mathrm{rot} \ [\mathrm{K}]$',
        r'$T_\mathrm{ex} \ [\mathrm{K}]$',
        r'$\log(N_\mathrm{p}) \ [\mathrm{cm^{-2}}]$',
        r'$\sigma_\mathrm{v} \ [\mathrm{km\, s^{-1}}]$',
        r'$f_\mathrm{o}$',
]

TEX_LABELS_NU = [  # without units
        r'$v_\mathrm{lsr}$',
        r'$T_\mathrm{rot}$',
        r'$T_\mathrm{ex}$',
        r'$\log(N_\mathrm{p})$',
        r'$\sigma_\mathrm{v}$',
        r'$f_\mathrm{o}$',
]


def get_par_names(ncomp=None):
    if ncomp is not None:
        return [
                f'{label}{n}'
                for label in ('v', 'Tk', 'Tx', 'N', 's', 'o')
                for n in range(1, ncomp+1)
        ]
    else:
        return ['v', 'Tk', 'Tx', 'N', 's', 'o']


cdef class AmmoniaSpectrum(Spectrum):
    cdef:
        double[::1] tbg_arr
        Transition trans

    def __init__(self, xarr, data, noise, trans_id=1):
        """
        Parameters
        ----------
        xarr : array
            Frequency axis array. Channels must be in ascending order.
            **units**: Hz
        data : array
            Brightness temperature intensity values.
            **units**: K
        noise : number
            The brightness temperature baseline RMS noise level.
            **units**: K
        trans_id : int
            para-NH3 meta-stable transition ID
                1 -> (1,1)
                2 -> (2,2)
                ...
                9 -> (9,9)
        """
        cdef long i
        assert trans_id in range(1, 10)
        super().__init__(xarr, data, noise, rest_freq=self.trans.nu,
                trans_id=trans_id)
        self.trans = TRANS[trans_id-1]
        # initialize background brightness temperature values
        self.tbg_arr = np.empty_like(data)
        for i in range(self.size):
            nu = self.xarr[i]
            T0 = H * nu / KB
            self.tbg_arr[i] = 1.0 / c_expm1(T0 / TCMB)


cdef inline double swift_convert(double tkin) nogil:
    """
    Convert a gas kinetic temperature to ammonia rotation temperature using the
    "cold ammonia" approximation derived in Swift et al. (2005) by equation A6.
        https://ui.adsabs.harvard.edu/abs/2005ApJ...620..823S/abstract
    """
    return tkin / (1.0 + (tkin / 41.18) * c_log(1.0 + 0.6 * c_exp(-15.7 / tkin)))


cdef inline double c_partition_level(long j, double trot) nogil:
    IF __APPROX:
        return (
                (2 * j + 1)
                * fast_expn(H * (BROT * j * (j + 1)
                + (CROT - BROT) * j * j) / (KB * trot))
        )
    ELSE:
        return (
                (2 * j + 1)
                * c_exp(-H * (BROT * j * (j + 1)
                + (CROT - BROT) * j * j) / (KB * trot))
        )


cdef inline double c_partition_func(bint para, double trot) nogil:
    # NOTE could likely interpolate for improved performance
    cdef:
        long j
        double Qtot = 0.0
    if para:
        for j in JPARA:
            Qtot += c_partition_level(j, trot)
    else:
        for j in JORTH:
            Qtot += 2 * c_partition_level(j, trot)
    return Qtot


def partition_level(j, trot):
    return c_partition_level(j, trot)


def partition_func(para, trot):
    return c_partition_func(para, trot)


DEF T0_LO = H * 23.0e9 / KB  # Frequency in Hz
DEF T0_HI = H * 28.0e9 / KB
DEF T0_XMIN = T0_LO / 8.0    # Upper Tex = 8.0 K
DEF T0_XMAX = T0_HI / 2.7    # Lower Tex = 2.7 K
DEF T0_SIZE = 1000
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


cdef void c_amm_predict(AmmoniaSpectrum s, double *params, long ndim,
            bint cold, bint lte) nogil:
    cdef:
        long i, j, k
        long ncomp = ndim // 6
        long nu_lo_ix, nu_hi_ix
        double trot, tex, ntot, sigm, voff, orth
        double zlev, qtot, species_frac, pop_rotstate
        double expterm, fracterm, widthterm, tau_main
        double hf_freq, hf_width, hf_offset, nu, T0, hf_tau_sum, tau_exp
        double hf_tau, hf_nucen, hf_idenom
        double nu_cutoff, nu_lo, nu_hi
        Transition t = s.trans
    for i in range(s.size):
        s.pred[i] = 0.0
    for i in range(ncomp):
        voff = params[        i]
        trot = params[  ncomp+i]
        tex  = params[2*ncomp+i]
        ntot = params[3*ncomp+i]
        sigm = params[4*ncomp+i]
        orth = params[5*ncomp+i]
        if cold:
            trot = swift_convert(trot)
        if lte:
            tex = trot
        # Calculate the partition function and the level populations
        zlev = c_partition_level(t.n, trot)
        qtot = c_partition_func(t.para, trot)
        # Calculate the main line optical depth
        species_frac = 1.0 - orth if t.para else orth
        pop_rotstate = 10.0**ntot * species_frac * zlev / qtot
        expterm = (
                (1.0 - c_exp(-H * t.nu / (KB * tex))) /
                (1.0 + c_exp(-H * t.nu / (KB * tex)))
        )
        fracterm = CCMS**2 * t.ea / (8 * M_PI * t.nu**2)
        widthterm = CKMS / (sigm * t.nu * c_sqrt(2 * M_PI))
        tau_main = pop_rotstate * fracterm * expterm * widthterm
        # Calculate the velocity/frequency related constants for the
        # hyperfine transitions.
        for j in range(s.size):
            s.tarr[j] = 0.0
        for j in range(t.nhf):
            # Hyperfine (HF) dependent values
            hf_freq   = (1.0 - t.voff[j] / CKMS) * t.nu
            hf_width  = sigm / CKMS * hf_freq
            hf_offset = voff / CKMS * hf_freq
            hf_nucen  = hf_freq - hf_offset
            hf_tau    = tau_main * t.tau_wts[j]
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
                for k in range(nu_lo_ix, nu_hi_ix):
                    nu = s.xarr[k] - hf_nucen
                    tau_exp = nu * nu * hf_idenom
                    s.tarr[k] += hf_tau * fast_expn(tau_exp)
            ELSE:
                for k in range(s.size):
                    nu = s.xarr[k] - hf_nucen
                    tau_exp = nu * nu * hf_idenom
                    s.tarr[k] += hf_tau * c_exp(-tau_exp)
        # Compute the brightness temperature
        for j in range(s.size):
            if s.tarr[j] == 0.0:
                continue
            T0 = H * s.xarr[j] / KB
            # Eq: (T0 / (exp(T0 / tex) - 1) - T0 / (exp(T0 / TCMB) - 1))
            #     * (1 - exp(-tau))
            IF __APPROX:
                s.pred[j] += (
                    T0 * (c_iemtex_interp(T0 / tex) - s.tbg_arr[j])
                    * (1.0 - fast_expn(s.tarr[j]))
                )
            ELSE:
                s.pred[j] += (
                    (T0 / c_expm1(T0 / tex) - T0 * s.tbg_arr[j])
                    * (1.0 - c_exp(-s.tarr[j]))
                )


def amm_predict(AmmoniaSpectrum s, double[::1] params, bint cold=False,
        bint lte=False):
    c_amm_predict(s, &params[0], params.shape[0], cold, lte)


cdef class AmmoniaRunner(Runner):
    cdef:
        bint cold, lte
        long n_spec
        AmmoniaSpectrum[:] spectra

    def __init__(self, spectra, utrans, ncomp=1, cold=False, lte=False):
        """
        Parameters
        ----------
        spectra : array(`AmmoniaSpectrum`)
            Array of spectrum wrapper objects
        utrans : `PriorTransformer`
            Prior transformer class that samples the prior from the unit cube
            for the five model ammonia parameters.
        ncomp : int, default 1
            Number of velocity components
        cold : bool, default False
            Apply the Swift et al. (2005) approximation to convert gas kinetic
            temperature to rotation temperature.
        lte : bool, default False
            Set the excitation temperature equal to the rotation temperature.

        Attributes
        ----------
        null_lnZ : number
            Natural log evidence for the "null model" of a constant equal to
            zero.
        run_lnZ : number
            Natural log global evidence from the MultiNest run.
        """
        cdef:
            AmmoniaSpectrum spec
        assert ncomp > 0
        self.n_model = 6
        self.spectra = spectra
        self.utrans = utrans
        self.ncomp = ncomp
        self.cold = cold
        self.lte = lte
        self.n_spec = len(spectra)
        self.n_params = self.n_model * ncomp
        self.ndim = self.n_params  # no nuisance parameters
        self.null_lnZ = 0.0
        self.n_chan_tot = 0
        for spec in self.spectra:
            self.null_lnZ += spec.null_lnZ
            self.n_chan_tot += spec.size
        self.run_lnZ = np.nan

    @classmethod
    def from_data(cls, spec_data, utrans, **kwargs):
        spectra = np.array([AmmoniaSpectrum(*args) for args in spec_data])
        return cls(spectra, utrans, **kwargs)

    cdef void c_loglikelihood(self, double *utheta, double *lnL):
        cdef:
            long i
            AmmoniaSpectrum spec
        lnL[0] = 0.0
        self.utrans.c_transform(utheta, self.ncomp)
        for i in range(self.n_spec):
            spec = self.spectra[i]
            c_amm_predict(spec, utheta, self.ndim, self.cold, self.lte)
            lnL[0] += spec.c_loglikelihood()

    def get_spectra(self):
        return np.array(self.spectra)

    def predict(self, double[::1] params):
        cdef:
            long i
            AmmoniaSpectrum spec
        if params.shape[0] != self.ndim:
            ncomp = self.ncomp
            shape = params.shape[0]
            raise ValueError(f'Invalid shape for ncomp={ncomp}: {shape}')
        for i in range(self.n_spec):
            spec = self.spectra[i]
            c_amm_predict(spec, &params[0], self.ndim, self.cold, self.lte)


##############################################################################
#                                 Tests
##############################################################################

# NOTE for these tests to pass, the module should be compiled with
# `__NEW_CONST=False`
def test_partition_level():
    # Values computed from `Zpara` and `Zortho` values in
    #   pyspeckit.spectrum.models.ammonia.ammonia_model
    tol = -7.0
    zlev1 = c_partition_level(1, 10.0)  # J=1, Trot=10.0
    zlev1_psk = 0.29279893434489096
    np.testing.assert_almost_equal(zlev1, zlev1_psk, decimal=7)
    zlev2 = c_partition_level(2, 10.0)  # J=2, Trot=10.0
    zlev2_psk = 0.007933862262432792
    np.testing.assert_almost_equal(zlev2, zlev2_psk, decimal=7)
    qpara = c_partition_func(True, 10.0)
    qpara_psk = 0.30073281405688107
    np.testing.assert_almost_equal(qpara, qpara_psk, decimal=7)


def test_swift_convert():
    tkin = 15
    trot = swift_convert(tkin)
    trot_psk = 14.023487575888257
    np.testing.assert_almost_equal(trot, trot_psk, decimal=8)


def test_iemtex_interp():
    fine_x = np.linspace(T0_XMIN, T0_XMAX, 100000)
    def f(x): return 1 / (np.exp(x) - 1)
    def diff(x): return np.abs((iemtex_interp(x) - f(x)) / f(x))
    diffs = np.array([diff(x) for x in fine_x])
    np.testing.assert_almost_equal(diffs.max(), 0, decimal=5)


def test_profile_predict(AmmoniaSpectrum s, double[::1] params, bint
        cold=False, bint lte=False, long n_repeat=1000):
    cdef:
        long i
    for i in range(n_repeat):
        # Add an offset to mess with the cache just a little
        params[0] += 1e-16
        c_amm_predict(s, &params[0], params.shape[0], cold, lte)
    params[0] -= n_repeat * 1e-16


