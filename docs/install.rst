Installation and Requirements
=============================
Nested Sampling in   ``nestfit`` is perfomed using `MultiNest
<https://github.com/farhanferoz/MultiNest/>`_, written by Farhan Feroz and
published in Feroz & Hobson
(`2008 <https://ui.adsabs.harvard.edu/abs/2008MNRAS.384..449F/abstract>`_)
and Feroz, Hobson, & Bridges
(`2009 <https://ui.adsabs.harvard.edu/abs/2009MNRAS.398.1601F/abstract>`_).
Therefore, MultiNest must first be compiled and placed in your host systems
shared library path. This must be done so that it can be linked against when
compiling the Cython extension module for ``nestfit``.

Installing MultiNest
--------------------
A fortran compiler, such as ``gfortran`` or ``ifort``, and LAPACK are necessary
to build MultiNest.  Parallelism in ``nestfit`` is implemented using Python's
``multiprocessing`` module, so it is not strictly necessary to compile
MultiNest against an MPI library (e.g., MPICH).  LAPACK should be included with
the anaconda distribution.  On a 64-bit Linux system using ``anaconda`` this
may be done with:

.. code-block:: bash

    $ conda install gfortran_linux-64 mpich

Then place one of the following lines into your local shell configuration file
(e.g., ``~/.bashrc`` or ``~/.profile``):

.. code-block:: bash

    # if installing into your base conda environment
    export LD_LIBRARY_PATH=<ANACONDA-PATH>/lib/:$LD_LIBRARY_PATH
    # or if installing into a specific conda environment
    export LD_LIBRARY_PATH=<ANACONDA-PATH>/envs/<ENV-NAME>/lib/:$LD_LIBRARY_PATH

Then, download the latest copy of MultiNest and enter the directory for the
latest version (replace ``<VERSION>`` with, e.g, ``v3.12``):

.. code-block:: bash

    $ git clone https://github.com/farhanferoz/MultiNest
    $ cd MultiNest/MultiNest_<VERSION>_CMake/multinest

MultiNest can then be built with ``cmake`` following the supplied installation
instructions (see the ``README`` file under "Building and installing MultiNest
CMake version" within the above directory):

.. code-block:: bash

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

MultiNest should now be compiled and ready for linking. As a final step, the
directory containing the ``bin/`` and ``lib/`` directories (the present
directory if following the above instructions) must be added as the shell
environment variable ``MNEST_DIR``:

.. code-block:: bash

    export MNEST_DIR=<MULTINEST-PATH>/MultiNest/MultiNest_<VERSION>_CMake/multinest

Now, either open a new terminal our ``source`` your shell configuration file.
It should now be possible to install ``nestfit``.

Installing NestFit
------------------
Due to rapid pace of current development, it is recommended for the time being
to clone development branch and build the module in-place. It is Python
best-practices to activate a virtual environment for this project (perhaps the
same conda environment possibly used previously for MultiNest), but this is not
strictly necessary.

.. code-block:: bash

    $ git clone https://github.com/autocorr/nestfit
    $ cd nestfit
    $ pip install -r requirements.txt
    $ make

If compilation was a success, the ``nestfit`` module may then be imported from
Python appending this directory to your ``PYTHON_PATH``. Congratulations!
Instructions on getting started may be found in the :doc:`quick-start guide
<quickstart>`.

Requirements
------------
The python dependencies can be found in ``requirements.txt``:

.. code-block:: none

    astropy
    cython
    getdist
    h5py
    matplotlib
    numpy
    photutils
    pyspeckit
    scipy
    spectral_cube

The ``matplotlib`` and ``getdist`` packages are only required for plotting;
``photutils`` is only required for sub-grid corrections when computing
information kernels.
