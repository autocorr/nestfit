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

License
=======
This software is licensed under the MIT license. A copy is supplied in the
LICENSE file. This software makes use of the ammonia model code found in
``pyspeckit`` by Adam Ginsburg & Jordan Mirocha and contributors. The
implementation in ``pyspeckit`` was originally based on an implementation by
Erik Rosolowsky described in `Rosolowsky et al. (2008) <https://ui.adsabs.harvard.edu/abs/2008ApJS..175..509R/abstract>`_.
