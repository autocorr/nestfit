nestfit
=======
Automatically decompose ammonia inversion transition spectra into multiple
velocity components using Nested Sampling. The number of components are
discriminated between using Bayes' factors computed from the evidences.  The
Nested Sampling is performed using
`MultiNest <https://github.com/farhanferoz/MultiNest/>`_
written by Farhan Feroz and the python wrapper
`PyMultiNest <https://github.com/JohannesBuchner/PyMultiNest>`_
written by Johannes Buchner.

Nested Sampling (NS) is a Monte Carlo technique, and thus frequently requires
many more evaluations of the likelihood function than non-linear least square
optimization techniques such as the Levenberg-Marquardt algorithm (LM).
Preliminary tests fitting a single ammonia velocity component with NS require
approximately two orders of magnitude more likelihood evaluations than LM (~50
versus ~3000). However, while NS comes with a substantial computational cost,
it has several important advantages not offered by non-linear least squares
algorithm:

- initial guesses are not required
- robust to highly multi-modal posterior distributions
- the full set of posteriors are returned for robust estimation of parameter
  uncertainties
- the evidence is computed for comparison between models
- priors are specified to condition the data on reasonable distributions within
  parameter space

The code has been optimized for numerical efficiency by implementing the
ammonia model prediction in a Cython extension module. Comparing to PySpecKit
v0.1.22 reference implementation as of August, 2019, this provides a factor of
104 times reduction in time to compute the predicted spectrum.


License
=======
This software is licensed under the MIT license. A copy is supplied in the
LICENSE file. This software makes use of the ammonia model code found in
``pyspeckit`` by Adam Ginsburg & Jordan Mirocha and contributors. The
implementation in ``pyspeckit`` was originally based on an implementation by
Erik Rosolowsky described in `Rosolowsky et al. (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJS..175..509R/abstract>`_.
