#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

include "model_includes.pxi"
include "array_sizes.pxi"

from nestfit.core.core cimport (Spectrum, Runner)


# Number of model parameters
DEF N_PARAMS = 3


cdef void c_gauss_predict(Spectrum s, double *params, int ndim) nogil:
    cdef:
        int i, j
        int nu_lo_ix, nu_hi_ix
        int ncomp = ndim // N_PARAMS
        double voff, sigm, peak
        double nu, nu_width, nu_cen, nu_denom
        double nu_cutoff, nu_lo, nu_hi
    for i in range(s.size):
        s.pred[i] = 0.0
    for i in range(ncomp):
        voff = params[        i]
        sigm = params[  ncomp+i]
        peak = params[2*ncomp+i]
        nu_width = sigm / CKMS * s.rest_freq
        nu_cen   = s.rest_freq * (1 - voff / CKMS)
        nu_denom = 0.5 / (nu_width * nu_width)
        # Gaussians are approximated by only computing them within the range
        # of `exp(-12.5)` (3.7e-6) away from the line center.
        #   Eq:  exp(-nu**2 * hf_idenom) = exp(-12.5)
        nu_cutoff = c_sqrt(12.5 / nu_denom)
        nu_lo = (nu_cen - s.nu_min - nu_cutoff)
        nu_hi = (nu_cen - s.nu_min + nu_cutoff)
        # Get the lower and upper indices then check bounds
        nu_lo_ix = <int>c_floor(nu_lo/s.nu_chan)
        nu_hi_ix = <int>c_floor(nu_hi/s.nu_chan)
        if nu_hi_ix < 0 or nu_lo_ix > s.size-1:
            continue
        nu_lo_ix = 0 if nu_lo_ix < 0 else nu_lo_ix
        nu_hi_ix = s.size-1 if nu_hi_ix > s.size-1 else nu_hi_ix
        # Calculate the Gaussian line profile over the interval
        for j in range(nu_lo_ix, nu_hi_ix):
            nu = s.xarr[j] - nu_cen
            s.pred[j] += peak * fast_expn(nu * nu * nu_denom)


def gauss_predict(Spectrum s, double[::1] params):
    c_gauss_predict(s, &params[0], params.shape[0])


cdef class GaussianRunner(Runner):
    cdef Spectrum spectrum

    def __init__(self, spectrum, utrans, ncomp=1):
        """
        Parameters
        ----------
        spectrum : Spectrum
            Spectrum wrapper object
        utrans : `PriorTransformer`
            Prior transformer class that samples the prior from the unit cube
            for the three model parameters.
        ncomp : int, default 1
            Number of velocity components.

        Attributes
        ----------
        null_lnZ : number
            Natural log evidence for the "null model" of a constant equal to
            zero.
        run_lnZ : number
            Natural log global evidence from the MultiNest run.
        """
        assert ncomp > 0
        self.n_model = N_PARAMS
        self.spectrum = spectrum
        self.utrans = utrans
        self.ncomp = ncomp
        self.n_spec = 1
        self.n_params = self.n_model * ncomp
        self.ndim = self.n_params  # no nuisance parameters
        self.null_lnZ = 0.0
        self.n_chan_tot = 0
        self.null_lnZ = spectrum.null_lnZ
        self.n_chan_tot += spectrum.size
        self.run_lnZ = np.nan

    @classmethod
    def from_data(cls, spec_data, utrans, **kwargs):
        return cls(Spectrum(*spec_data), utrans, **kwargs)

    cdef void c_loglikelihood(self, double *utheta, double *lnL):
        lnL[0] = 0.0
        self.utrans.c_transform(utheta, self.ncomp)
        c_gauss_predict(self.spectrum, utheta, self.ndim)
        lnL[0] += self.spectrum.c_loglikelihood()

    def get_spectrum(self):
        return np.array(self.spectrum)

    def predict(self, double[::1] params):
        if params.shape[0] != self.ndim:
            ncomp = self.ncomp
            shape = params.shape[0]
            raise ValueError(f'Invalid shape for ncomp={ncomp}: {shape}')
        c_gauss_predict(self.spectrum, &params[0], self.ndim)


# Aliases and metadata for external use at module level scope
N = N_PARAMS
NAME = 'gaussian'
model_predict = gauss_predict
ModelSpectrum = Spectrum
ModelRunner = GaussianRunner

PAR_NAMES = ['voff', 'sigm', 'peak']
PAR_NAMES_SHORT = ['v', 's', 'pk']

TEX_LABELS = [
        r'$v_\mathrm{lsr}$',
        r'$\sigma_\mathrm{v}$',
        r'$T_\mathrm{pk}$',
]

TEX_LABELS_WITH_UNITS = [
        r'$v_\mathrm{lsr} \ [\mathrm{km\, s^{-1}}]$',
        r'$\sigma_\mathrm{v} \ [\mathrm{km\, s^{-1}}]$',
        r'$T_\mathrm{pk} \ [\mathrm{K}]$',
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


