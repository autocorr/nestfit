nestfit
=======
This repository provides software tools to automatically decompose ammonia
inversion transition spectra into multiple velocity components. The number of
components is selected using Bayes' factors and evidences computed using Nested
Sampling. This is currently a work in progress and being used in analysis of
GBT/VLA ammonia observations of infrared dark clouds (Svoboda et al., in
prep.).

The Nested Sampling is performed using
`MultiNest <https://github.com/farhanferoz/MultiNest/>`_
written by Farhan Feroz and wrapped directly with Cython.

Nested Sampling
---------------
Nested Sampling (NS) is a Monte Carlo technique principally aimed at
calculating the Bayesian evidence. In general NS requires many more evaluations
of the likelihood function than typical non-linear least squares optimization
techniques, such as the Levenberg-Marquardt algorithm (LM).  Preliminary tests
fitting a single ammonia velocity component with NS require approximately two
orders of magnitude more likelihood evaluations than LM (~50 versus ~3000).
However, while NS may come with a substantial computational cost, it has
several advantages over non-linear least squares algorithms:

- Parameter initial guesses are not required.
- Explorations of highly multi-modal and covariant posterior distributions are
  possible.
- The full set of posterior samples are returned for robust estimation of
  parameter uncertainties.
- The evidence is computed for model comparison, and the statistical
  uncertainty on the evidence may be computed from the results of a single run.
- A clear stopping criteria is provided based on the convergence of the
  evidence.
- Priors are specified to condition the data on reasonable distributions within
  parameter space.
- No "burn in" run required by many Markov Chain Monte Carlo methods.

To make fitting large data-cubes (>10,000 spectra) computationally tractable,
the numeric routines have been implemented in an optimized Cython extension
module.  As of August, 2019, this provides a factor of 220 reduction in time to
compute a model spectrum compared to the PySpecKit v0.1.22 reference
implementation.

Installation and Requirements
-----------------------------
To use this package MultiNest must first be compiled and placed in your host
systems shared library path. This must be done so that it can be linked against
when compiling the Cython extension module. A fortran compiler is necessary to
build MultiNest, such as `gfortran` or `ifort`.  To use the MPI
functionality, an MPI library is also required, such as MPICH. If using
`anaconda` on 64-bit Linux this could be done with:

.. codeblock:: shell

    $ conda install gfortran_linux-64 mpich

Then place one of the following lines into your shell configuration file:

.. codeblock:: shell

    $ # if installed into your base conda environment
    $ export LD_LIBRARY_PATH=<ANACONDA-PATH>/lib/:$LD_LIBRARY_PATH
    $ # or if installed into a specific conda environment
    $ export LD_LIBRARY_PATH=<ANACONDA-PATH>/envs/<ENV-NAME>/lib/:$LD_LIBRARY_PATH

MultiNest can then be built with `cmake` following the supplied installation
instructions:

.. codeblock:: shell

    $ cd MultiNest/build
    $ cmake ..
    $ make


License
=======
This software is licensed under the MIT license. A copy is supplied in the
LICENSE file.

This software makes use of the ammonia model code found in
``pyspeckit`` by Adam Ginsburg & Jordan Mirocha and contributors. The
implementation in ``pyspeckit`` was originally based on an implementation by
Erik Rosolowsky described in
`Rosolowsky et al. (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJS..175..509R/abstract>`_.

The implementation of the fast exponential function is taken from the LIME
radiative transfer code developed by Christian Brinch and the LIME development
team. These modified files ("fastexp.h", "fastexp.c") are licensed under the
GNU GPL v3.
