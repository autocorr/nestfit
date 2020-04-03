nestfit
=======
This repository provides software tools to automatically decompose ammonia
inversion transition spectra into multiple velocity components. The number of
components is selected using Bayes' factors and evidence values computed using
Nested Sampling. This library is currently under development and is being used
to analyze GBT/VLA ammonia observations of infrared dark clouds (Svoboda et
al., in prep.).

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

Installation and Requirements
-----------------------------
To use this package MultiNest must first be compiled and placed in your host
systems shared library path. This must be done so that it can be linked against
when compiling the Cython extension module. A fortran compiler, such as
``gfortran`` or ``ifort``, is necessary to build MultiNest.  To use the MPI
functionality, an MPI library is also required, such as MPICH. MPI however is
not necessary for use with ``nestfit`` as parallelism is implemented with
Python's ``multiprocessing`` module. If using ``anaconda`` on 64-bit Linux this
may be done with:

.. code-block:: bash

    $ conda install gfortran_linux-64 mpich

Then place one of the following lines into your shell configuration file:

.. code-block:: bash

    $ # if installed into your base conda environment
    $ export LD_LIBRARY_PATH=<ANACONDA-PATH>/lib/:$LD_LIBRARY_PATH
    $ # or if installed into a specific conda environment
    $ export LD_LIBRARY_PATH=<ANACONDA-PATH>/envs/<ENV-NAME>/lib/:$LD_LIBRARY_PATH

MultiNest can then be built with ``cmake`` following the supplied installation
instructions:

.. code-block:: bash

    $ cd MultiNest/build
    $ cmake ..
    $ make

Due to current rapid development, for the time being it is recommended to clone
this repository and build the module in-place. It may also be wise to activate
a virtual environment for this project, but this is not strictly necessary.

.. code-block:: bash

    $ git clone https://github.com/autocorr/nestfit
    $ cd nestfit
    $ make

If compilation was a success, the ``nestfit`` module may then be imported from
Python if this directory is in your ``PYTHON_PATH``.

License
=======
This software is licensed under the MIT license. A copy is supplied in the
LICENSE file.

This software makes use of the ammonia model code found in ``pyspeckit`` by
Adam Ginsburg & Jordan Mirocha and contributors. The Python implementation in
``pyspeckit`` was originally based on the IDL program ``nh3fit`` by Erik
Rosolowsky described in
`Rosolowsky et al. (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJS..175..509R/abstract>`_.

The implementation of the fast exponential function is taken from the LIME
radiative transfer code developed by Christian Brinch and the LIME development
team. These modified files ("fastexp.h", "fastexp.c") are licensed under the
GNU GPL v3.
