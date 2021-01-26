nestfit
=======
.. image:: https://readthedocs.org/projects/nestfit/badge/?version=latest
   :target: https://nestfit.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4470028.svg
   :target: https://doi.org/10.5281/zenodo.4470028

Welcome to the NestFit source code repository.  NestFit is a Bayesian framework
for fitting spectral line data containing multiple velocity components. The
official documentation and installation instructions may be found at
https://nestfit.readthedocs.io as well as in the included ``docs/`` directory.

If you make use of NestFit or a derivative in an academic work, please cite the
forth-coming publication (Svoboda in prep.) or this Zenodo DOI:

.. code::

    @software{brian_svoboda_2021_4470028,
    author       = {Brian Svoboda},
    title        = {autocorr/nestfit: Initial public release},
    month        = jan,
    year         = 2021,
    publisher    = {Zenodo},
    version      = {v0.2},
    doi          = {10.5281/zenodo.4470028},
    url          = {https://doi.org/10.5281/zenodo.4470028}
    }


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
