cimport cython

import numpy as np
cimport numpy as np
np.import_array()

from nestfit.core.math cimport (M_PI, c_exp, c_expm1, c_sqrt, c_floor,
        c_log, fast_expn, calcExpTableEntries, fillErfTable)


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


