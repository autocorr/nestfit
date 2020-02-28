#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

cimport cython
from cymem.cymem cimport Pool

import numpy as np
cimport numpy as np
np.import_array()

from nestfit.cmultinest cimport run as c_run_multinest


cdef class PriorTransformer:
    cdef:
        int npriors
        Pool mem
        void **priors
        list __priors

    def __cinit__(self, priors):
        self.mem = None
        self.npriors = 0
        self.priors = NULL
        self.__priors = None

    def __init__(self, priors):
        """
        Evaluate the prior transformation functions on the unit cube. The
        `.transform` method is passed to MultiNest and called on each
        likelihood evaluation.

        Parameters
        ----------
        priors : iterable(Prior)
            List-like of individual `Prior` instances.
        """
        self.mem = Pool()
        self.npriors = len(priors)
        self.__priors = priors
        for prior in priors:
            assert isinstance(prior, Prior)
        # NOTE Cython does not allow pointers to object types, which is what
        # the Cython extension type `Prior` is, so we need to do some black
        # magic hackery with void* and casting to make it work.
        self.priors = <void**>self.mem.alloc(self.npriors, sizeof(void*))
        for i, prior in enumerate(self.__priors):
            self.priors[i] = <void*>(prior)

    cdef void transform(self, double *utheta, int ncomp) nogil:
        """
        Parameters
        ----------
        utheta : double*
            Pointer to parameter unit cube.
        ncomp : int
            Number of components. `utheta` should have dimension [5*n].
        """
        # NOTE may do unsafe writes if `utheta` does not have the same
        # size as the number of components `ncomp`.
        cdef int i
        for i in range(self.npriors):
            (<Prior> self.priors[i]).interp(utheta, ncomp, i)


