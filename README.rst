nestfit
=======
Welcome to the NestFit source code repository.  NestFit is a Bayesian framework
for fitting spectral line data containing multiple velocity components. The
official documentation and installation instructions may be found at
https://nestfit.readthedocs.io as well as in the included ``docs/`` directory.

If you make use of NestFit in an academic work, please cite the forth-coming
publication that will be listed here (Svoboda `in prep.`).


License
=======
This software is licensed under the MIT license. A copy is supplied in the
LICENSE file.

This software makes use of the ammonia model code found in ``pyspeckit`` by
Adam Ginsburg & Jordan Mirocha and contributors. The Python implementation in
``pyspeckit`` was originally based on the IDL program ``nh3fit`` by Erik
Rosolowsky, described in
`Rosolowsky et al. (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJS..175..509R/abstract>`_.

The implementation of the fast exponential function is taken from the LIME
radiative transfer code developed by Christian Brinch and the LIME development
team. These modified files (``fastexp.h``, ``fastexp.c``) are licensed under
the GNU GPL v3.
