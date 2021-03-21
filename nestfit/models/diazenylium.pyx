#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

include "model_includes.pxi"
include "array_sizes.pxi"

from nestfit.core.core cimport (Transition, HyperfineSpectrum, Runner)
from nestfit.models.hyperfine cimport c_hf_predict


# Number of rotational levels: J=1 to J=3
DEF N_LEVELS = 3
# Number of model parameters
DEF N_PARAMS = 4


# Initialize statically allocated C arrays with data values for each
# transition. Values taken from:
#   `pyspeckit.pyspeckit.spectrum.models.n2hp`
cdef:
    long i
    long NHF[N_LEVELS]
    double NU[N_LEVELS]
    double VOFF[N_LEVELS][MAX_HF_N]
    double TAU_WTS[N_LEVELS][MAX_HF_N]


# Number of hyperfine satellite lines per rotational transition
NHF = [
        15,  # (1-0)
        40,  # (2-1)
        45,  # (3-2)
]

# Rest frequencies in Hz
NU = [
        93173.7637e6,   # (1-0)
        186344.8420e6,  # (2-1)
        279511.8325e6,  # (3-2)
]

# Velocity offsets of the hyperfine transitions in km/s
VOFF[0][:NHF[0]] = [  # (1-0)
    -7.9930, -7.9930, -7.9930, -0.6112, -0.6112, -0.6112,  0.0000,  0.9533,
     0.9533,  5.5371,  5.5371,  5.5371,  5.9704,  5.9704,  6.9238,
]
VOFF[1][:NHF[1]] = [  # (2-1)
    -4.6258, -4.5741, -4.4376, -4.2209, -4.0976, -3.8808, -3.1619, -2.9453,
    -2.3469, -1.9290, -1.5888, -1.5516, -1.4523, -1.1465, -0.8065, -0.6532,
    -0.4694, -0.1767,  0.0000,  0.0071,  0.1137,  0.1291,  0.1617,  0.2239,
     0.5237,  0.6384,  0.7405,  2.1394,  2.5158,  2.5444,  2.6225,  2.8844,
     3.0325,  3.0990,  3.2981,  3.5091,  3.8148,  3.8201,  6.9891,  7.5057,
]
VOFF[2][:NHF[2]] = [  # (3-2)
    -3.0666, -2.9296, -2.7221, -2.6563, -2.5270, -2.4010, -2.2535, -2.1825,
    -2.1277, -1.5862, -1.0158, -0.6131, -0.6093, -0.5902, -0.4872, -0.4725,
    -0.2757, -0.0697, -0.0616, -0.0022,  0.0000,  0.0143,  0.0542,  0.0561,
     0.0575,  0.0687,  0.1887,  0.2411,  0.3781,  0.4620,  0.4798,  0.5110,
     0.5540,  0.7808,  0.9066,  1.6382,  1.6980,  2.1025,  2.1236,  2.1815,
     2.5281,  2.6458,  2.8052,  3.0320,  3.4963,
]

# Optical depth weights of the hyperfine transitions, unitless and normalized.
# Weights are taken from pyspeckit.
TAU_WTS[0][:NHF[0]] = [  # (1-0)
    0.025957, 0.065372, 0.019779, 0.004376, 0.034890, 0.071844, 0.259259,
    0.156480, 0.028705, 0.041361, 0.013309, 0.056442, 0.156482, 0.028705,
    0.037038,
]
TAU_WTS[1][:NHF[1]] = [  # (2-1)
    0.008272, 0.005898, 0.031247, 0.013863, 0.013357, 0.010419, 0.000218,
    0.000682, 0.000152, 0.001229, 0.000950, 0.000875, 0.002527, 0.000365,
    0.000164, 0.021264, 0.031139, 0.000576, 0.200000, 0.001013, 0.111589,
    0.088126, 0.142604, 0.011520, 0.027608, 0.012800, 0.066354, 0.013075,
    0.003198, 0.061880, 0.004914, 0.035879, 0.011026, 0.039052, 0.019767,
    0.004305, 0.001814, 0.000245, 0.000029, 0.000004,
]
TAU_WTS[2][:NHF[2]] = [  # (3-2)
    0.001845, 0.001818, 0.003539, 0.014062, 0.011432, 0.000089, 0.002204,
    0.002161, 0.000061, 0.000059, 0.000212, 0.000255, 0.000247, 0.000436,
    0.010208, 0.000073, 0.007447, 0.000000, 0.000155, 0.000274, 0.174603,
    0.018683, 0.135607, 0.100527, 0.124866, 0.060966, 0.088480, 0.001083,
    0.094510, 0.014029, 0.007191, 0.022222, 0.047915, 0.015398, 0.000071,
    0.000794, 0.001372, 0.007107, 0.016618, 0.009776, 0.000997, 0.000487,
    0.000069, 0.000039, 0.000010,
]


# Fill global transition struct array with values allocated above.
cdef:
    Transition[N_LEVELS] TRANS
for i in range(N_LEVELS):
    TRANS[i].n    = i + 1
    TRANS[i].para = False
    TRANS[i].nu   = NU[i]
    TRANS[i].ea   = np.nan
    TRANS[i].nhf  = NHF[i]
    TRANS[i].voff = VOFF[i]
    TRANS[i].tau_wts = TAU_WTS[i]


cdef class DiazenyliumSpectrum(HyperfineSpectrum):
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
            NH3 meta-stable transition ID.
                1 -> (1-0)
                2 -> (2-1)
                3 -> (3-2)
        """
        cdef:
            long i
            double nu
        assert trans_id in range(1, N_LEVELS+1)
        super().__init__(xarr, data, noise, rest_freq=self.trans.nu,
                trans_id=trans_id)
        self.trans = TRANS[trans_id-1]
        # initialize background brightness temperature values
        self.tbg_arr = np.empty_like(data)
        for i in range(self.size):
            nu = self.xarr[i]
            T0 = H * nu / KB
            self.tbg_arr[i] = 1.0 / c_expm1(T0 / TCMB)


cdef void c_nnhp_predict(DiazenyliumSpectrum s, double *params,
            long ndim) nogil:
    cdef:
        long i
        long ncomp = ndim // N_PARAMS
        double voff, tex, ltau, sigm
        Transition t = s.trans
    for i in range(s.size):
        s.pred[i] = 0.0
    for i in range(ncomp):
        voff = params[        i]
        tex  = params[1*ncomp+i]
        ltau = params[2*ncomp+i]
        sigm = params[3*ncomp+i]
        c_hf_predict(s, voff, tex, ltau, sigm)


def nnhp_predict(DiazenyliumSpectrum s, double[::1] params):
    c_nnhp_predict(s, &params[0], params.shape[0])


cdef class DiazenyliumRunner(Runner):
    cdef DiazenyliumSpectrum[:] spectra

    def __init__(self, spectra, utrans, ncomp=1):
        """
        Parameters
        ----------
        spectra : array(`DiazenyliumSpectrum`)
            Array of spectrum wrapper objects
        utrans : `PriorTransformer`
            Prior transformer class that samples the prior from the unit cube
            for the four model parameters.
        ncomp : int, default 1
            Number of velocity components

        Attributes
        ----------
        null_lnZ : number
            Natural log evidence for the "null model" of a constant equal to
            zero.
        run_lnZ : number
            Natural log global evidence from the MultiNest run.
        """
        cdef:
            DiazenyliumSpectrum spec
        assert ncomp > 0
        self.n_model = N_PARAMS
        self.spectra = spectra
        self.utrans = utrans
        self.ncomp = ncomp
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
        spectra = np.array([DiazenyliumSpectrum(*args) for args in spec_data])
        return cls(spectra, utrans, **kwargs)

    cdef void c_loglikelihood(self, double *utheta, double *lnL):
        cdef:
            long i
            DiazenyliumSpectrum spec
        lnL[0] = 0.0
        self.utrans.c_transform(utheta, self.ncomp)
        for i in range(self.n_spec):
            spec = self.spectra[i]
            c_nnhp_predict(spec, utheta, self.ndim)
            lnL[0] += spec.c_loglikelihood()

    def get_spectra(self):
        return np.array(self.spectra)

    def predict(self, double[::1] params):
        cdef:
            long i
            DiazenyliumSpectrum spec
        if params.shape[0] != self.ndim:
            ncomp = self.ncomp
            shape = params.shape[0]
            raise ValueError(f'Invalid shape for ncomp={ncomp}: {shape}')
        for i in range(self.n_spec):
            spec = self.spectra[i]
            c_nnhp_predict(spec, &params[0], self.ndim)


# Aliases and metadata for external use at module level scope
N = N_PARAMS
NAME = 'diazenylium'
model_predict = nnhp_predict
ModelSpectrum = DiazenyliumSpectrum
ModelRunner = DiazenyliumRunner

PAR_NAMES = ['voff', 'tex', 'ltau', 'sigm']
PAR_NAMES_SHORT = ['v', 'Tx', 'lt', 's']

TEX_LABELS = [
        r'$v_\mathrm{lsr}$',
        r'$T_\mathrm{ex}$',
        r'$\log(\tau_0)$',
        r'$\sigma_\mathrm{v}$',
]

TEX_LABELS_WITH_UNITS = [
        r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$',
        r'$T_\mathrm{ex} \ [\mathrm{K}]$',
        r'$\log(\tau_0)$',
        r'$\sigma_\mathrm{v} \ [\mathrm{km\, s^{-1}}]$',
]


def get_par_names(ncomp=None):
    if ncomp is not None:
        return [
                f'{label}{n}'
                for label in PAR_NAMES_SHORT
                for n in range(1, ncomp+1)
        ]
    else:
        return PAR_NAMES_SHORT


##############################################################################
#                                 Tests
##############################################################################

def test_profile_predict(DiazenyliumSpectrum s, double[::1] params,
        long n_repeat=1000):
    cdef:
        long i
    for i in range(n_repeat):
        # Add an offset to mess with the cache just a little
        params[0] += 1e-16
        c_nnhp_predict(s, &params[0], params.shape[0])
    params[0] -= n_repeat * 1e-16


