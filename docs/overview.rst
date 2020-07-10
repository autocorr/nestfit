Overview
========

Nested Sampling
---------------
Nested Sampling (NS) is a Monte Carlo technique principally aimed at
calculating the Bayesian evidence. In general NS requires many more evaluations
of the likelihood function than typical non-linear least squares optimization
techniques, such as the Levenberg-Marquardt algorithm (LM).  Preliminary tests
fitting a single ammonia velocity component with NS require approximately two
orders of magnitude more likelihood evaluations than LM (~50 versus ~5000).
However, while NS may come with a substantial computational cost, it has
several advantages over non-linear least squares algorithms:

- Explorations of highly multi-modal and covariant posterior distributions are
  possible.
- The full set of posterior samples are returned for robust estimation of
  parameter uncertainties.
- The evidence is computed for model comparison, and the statistical
  uncertainty on the evidence may be computed from the results of a single run.
- A clear stopping criteria is provided based on the convergence of the
  evidence.
- Parameter initial guesses are not required.
- Priors are specified to condition the data on reasonable distributions within
  parameter space.
- No "burn in" run required by many Markov Chain Monte Carlo methods.

To make fitting large data-cubes (>10,000 spectra) computationally tractable,
the numeric routines have been implemented in an optimized Cython extension
module.  As of March, 2020, this provides a factor of 450 reduction in time to
compute a model spectrum compared to the PySpecKit v0.1.22 reference
implementation. Please note that up-to-date physical constants and
spectroscopic constants are used and these lead to a 1% deviation from the
model in PySpecKit.


