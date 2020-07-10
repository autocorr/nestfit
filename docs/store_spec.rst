========================
Store-file specification
========================

Directory Description
---------------------
Model fit products are stored in specified way in a hierarchical HDF5 file. A
store may be opened using an instance of the ``nestfit.HdfStore`` class. Some
functionality is encapsulated using this class, but most data operations take
place directly on the primary ``h5py.File`` instance. Please see the ``h5py``
`documentation <https://h5py.readthedocs.io>`_ for a description of HDF5 files
and how to use them.

For parallel cube-fitting, multiple HDF5 files in a special directory are used
such that each process may write without locking.  The directory for the store
has the extension ``.store`` and the following structure:

.. code-block :: none

    - <NAME>.store
        - chunk0.hdf
        - chunk1.hdf
        - ...
        - table.hdf

where the files ``chunk<N>.hdf`` are the HDF5 files for each process. Once a
cube has been fit, the entries within each chunk are soft-linked to the
single file ``table.hdf`` without copying. For more information on HDF5 soft
links, see the ``h5py`` documentation `here
<https://h5py.readthedocs.io/en/stable/high/group.html#group-softlinks>`_.
The ``table.hdf`` file stores the metadata, cube header, and the aggregated
products created from post-processing.

The specification and layout of the data in the ``table.hdf`` is given in the
following section. The HDF5 file may be accessed directly with the attribute
``HdfStore.hdf``, which is an instance of ``h5py.File``.  Note that data
product arrays are strided in the C convention with fastest varying index being
furthest to the right. They are optimized for displaying maps of a given
parameter combination.


Specification
-------------

The data stored in the HDF table file has the following specification. Group
names are indicated by a ``"*"``, attributes by a ``"-"``, and datasets by a ``"="``
followed by the dimension. Child items are indicated by indentation. Group and
dataset names can be joined by a ``"/"``, so a valid path to the dataset
``posteriors`` in the group ``'/pix/0/0/1'`` would be
``hdf['/pix/0/0/1/posteriors']``, for example. Attributes are accessed with
``group.attrs['<NAME>']``.

.. code-block :: none

    * / : root group
    - lnZ_threshold: evidence threshold used when selecting one model over another
    - multinest_kwargs : additional keyword arguments passed to MultiNest
    - n_max_components : the maximum number of components to iteratively fit
    - naxis1 : the number of longitude pixels
    - naxis2 : the number of latitude pixels
    - nchunks : the number of HDF chunk files in the store
        * pix : hierarchical directory containing data for each pixel
            * <LON> : the longitude pixel number
                * <LAT> : the latitude pixel number
                - nbest : best fit model number
                - i_lon : longitude pixel number
                - i_lat : latitude pixel number
                    * <N> : model number
                    - AIC
                    - AICc
                    - BIC
                    - global_lnZ
                    - global_lnZ_err
                    - marg_cols
                    - marg_quantiles
                    - max_loglike
                    - max_loglike
                    - n_chan_tot
                    - n_live
                    - n_params
                    - n_samples
                    - ncomp
                    - null_lnZ
                    - par_names
                    = bestfit_params (n=1; p*m)
                    = map_params     (n=1; p*m)
                    = marginals      (n=2; M, p*m)
                    = posteriors     (n=2; n, p*m+2)
        * products : post-processing aggregate products
            = nbest                (n=2; b, l)
            = evidence             (n=3; m, b, l)
            = evidence_err         (n=3; m, b, l)
            = AIC                  (n=3; m, b, l)
            = AICc                 (n=3; m, b, l)
            = BIC                  (n=3; m, b, l)
            = conv_evidence        (n=3; m, b, l)
            = conv_nbest           (n=2; b, l)
            = marg_quantiles       (n=1; M)
            = nbest_MAP            (n=4; m, p, b, l)
            = nbest_bestfit        (n=4; m, p, b, l)
            = nbest_marginals      (n=5; m, p, M, b, l)
            = pdf_bins             (n=2; p, h)
            = post_pdfs            (n=6; r, m, p, h, b, l)
            = conv_post_pdfs       (n=6; r, m, p, h, b, l)
            = conv_marginals       (n=6; r, m, p, M, b, l)
            = peak_intensity       (n=4; t, m, b, l)
            = integrated_intensity (n=4; t, m, b, l)
            = hf_deblended         (n=5; t, m, S, b, l)
        * full_header : all header keywords stored as attributes
        * simple_header : subset of coordinate system related header keywords

    Product dimension key codes:
      * n: number of samples
      * b: latitude pixel index
      * l: longitude pixel index
      * p: model parameter
      * m: model component number
      * M: marginal distribution quantile
      * r: run number (ie, the index for the 1-comp run, 2-comp run, etc.)
      * h: marginal PDF bin
      * t: transition
      * S: spectral channel

    Quantile indices for marginal cubes:
      *  0 : 0.00   (min)
      *  1 : 0.01
      *  2 : 0.10
      *  3 : 0.25
      *  4 : 0.50   (median)
      *  5 : 0.75
      *  6 : 0.90
      *  7 : 0.99
      *  8 : 1.00   (max)
      *  9 : 0.1587 (-1 sigma) -- NOTE listed precision is truncated
      * 10 : 0.8413 (+1 sigma)
      * 11 : 0.0228 (-2 sigma)
      * 12 : 0.9772 (+2 sigma)
      * 13 : 0.0013 (-3 sigma)
      * 14 : 0.9987 (+3 sigma)

