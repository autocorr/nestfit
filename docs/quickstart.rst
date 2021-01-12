Quick-start guide
=================
This page provides a quick-start guide to using and running NestFit on a real
dataset. The ammonia (NH3) data fit are taken from the GBT KEYSTONE survey
published by `Keown et al. (2020)
<https://ui.adsabs.harvard.edu/abs/2019ApJ...884....4K/abstract>`_ and may be
downloaded from `CADC <https://www.canfar.net>`_, although access requires
registration.

A detailed and complete script for processing the KEYSTONE data may be
found in `this <https://github.com/autocorr/keystone_nestfit>`_ GitHub
repository. Various "book-keeping" functions are used to handle the spectral
cubes, noise maps, and the priors, but the general procedure may be directly
adapted to other ammonia datasets.

Applying NestFit to a dataset may be broadly broken down into four steps:

    1. Read data cubes and noise maps into the `CubeStack` class.
    2. Create the prior probability distributions and initialize the
       prior transformation class, `PriorTransformer`.
    3. Run the Bayesian inference with the `CubeFitter` class! This step
       produces a store file containing the posteriors and evidences per
       pixel per model per component.
    4. Post-process the store to aggregate per-pixel quantities into dense,
       multi-dimensional data products.

First import the relevant modules and classes:

.. code-block:: python

    import nestfit as nf
    from nestfit import (
            Distribution,
            Prior,
            ResolvedPlacementPrior,
            ConstantPrior,
            PriorTransformer,
    )

Prior probability distributions are central to the process of Bayesian
inference and should be carefully considered for the problem at hand.
Distributions are handled numerically and use interpolation when sampling
values. Thus the user is required to supply "x" and "y" values for the prior
PDF for each parameter. Note that it is best to avoid large ranges of zeros on
the lower and upper bounds of the distributions. The `ResolvePlacementPrior`
spaces velocity components apart from other based on the sampled centroids and
linewidths.

.. code-block:: python

    # The user must supply the "x" and "y" array of values of the prior PDF for
    # each parameter.
    d_voff = Distribution(x_voff, y_voff)
    d_trot = Distribution(x_trot, y_trot)
    d_tex  = Distribution(x_tex,  y_tex)
    d_ntot = Distribution(x_ntot, y_ntot)
    d_sigm = Distribution(x_sigm, y_sigm)
    # The numeric second parameter indicates the parameter of the model for
    # the constant-excitation ammonia model, these are:
    #   0 - "voff" velocity centroid
    #   1 - "trot" rotation temperature
    #   2 - "tex"  excitation temperature
    #   3 - "ntot" total molecular column density
    #   4 - "sigm" velocity dispersion (sigma)
    #   5 - "orth" ortho-to-total fraction
    priors = np.array([
            ResolvedPlacementPrior(
                Prior(d_voff, 0),
                Prior(d_sigm, 4),
                scale=1.2,
            ),
            Prior(d_trot, 1),
            Prior(d_tex,  2),
            Prior(d_ntot, 3),
            ConstantPrior(0, 5),
    ])
    utrans = PriorTransformer(priors)

.. code-block:: python

    cubes = [
            nf.DataCube('11_cube.fits', '11_rms.fits', trans_id=1),
            nf.DataCube('22_cube.fits', '22_rms.fits', trans_id=2),
    ]
    stack = nf.CubeStack(cubes)

With our data and priors properly initialized, we are ready to run NestFit!
Most of the important run-time parameters are set when initializing the
`CubeFitter` class. Keyword arguements taken by MultiNest are passed with a
dictionary from the `mn_kwargs` parameter. In this example the number of live
points is set to 500. Further live points may be added in the fitting process
based on the SNR of the data based on a multiplicative factor set with the
`nlive_snr_fact` parameter. In this example, the number of live points is
`nlive = 500 + 20 * SNR`. A larger number of live points ensures adequate
posterior sampling of faint secondary spectral components in the vicinity of
bright primary components.  For a parallel run, set `nproc` to the desired
number of threads. Setting `nproc` equal to 1 will run the cube fitting in
serial mode, which is desirable for debugging purposes.

.. code-block:: python

    store_name = f'run/test'
    runner_cls = nf.AmmoniaRunner
    fitter = nf.CubeFitter(stack, utrans, runner_cls, ncomp_max=2,
            mn_kwargs={'nlive': 500}, nlive_snr_fact=20)
    fitter.fit_cube(store_name=store_name, nproc=8)

Lastly, the post-processing steps may be run to aggregate the hierarchically
stored values computed in the run into densely stored data products to be
analyzed.

.. code-block:: python

    store = nf.HdfStore(store_name)
    nf.postprocess_run(store, stack, runner, par_bins=None, evid_kernel=None,
        post_kernel=None)


