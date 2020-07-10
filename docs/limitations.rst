===============================
Limitations and common problems
===============================

Spectrum limitations
--------------------
Please note that in the current implementation there are several limitations
and that these may cause difficult to diagnose bugs.

* The spectral axis must be in frequency units of Hertz
* The intensity must be in brightness temperature units of Kelvin
* The spectral axis must be in increasing order (low to high frequency)
* The spectrum must be contiguous and have uniform channel size (at least currently)

Priors
------
Always test your priors by making sure that transforming a unit-cube value of
all zeros or all ones returns a set of valid parameter values. A common but
difficult to debug error occurs when the prior transformer inserts spurious or
NaN values. The ``Distribution`` class uses a linear interpolation to invert
the CDF into a PPF, and this can be a source of errors if there are large
ranges where the PDF is exactly equal to zero.

Native code
-----------
For performance, the Cython extension module is compiled with aggressive
optimization flags, including ones that compile the shared object file
specifically for the native CPU architecture of the host system (the
``-march=native`` and ``-mtune=native`` flags). This means that NestFit must be
compiled on the host that the execution is to be performed on, or at least
machines that have the same CPU. This can lead to problems if NestFit is
compiled on your local work station, installed into your Python distribution on
a shared network filesystem, and then executed on a separate machine, such as a
computing cluster node. In this situation, the extension module needs to be
re-compiled (go to the package directory and run ``make``).

Misc common problems
--------------------
Please verify that the frequency axis contains the LSR centroid velocities
given by the prior. For example, if the prior contains velocities (-3, +3)
km/s, but the spectrum ranges over (40, 80) km/s, then the code will return a
blank spectrum. To avoid this, make sure to include the systemic velocity in
the relevant prior.
