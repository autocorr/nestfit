#cython: language_level=3


cdef extern from 'multinest.h':
    void run(
            bint IS,
            bint mmodal,
            bint ceff,
            int nlive,
            double tol,
            double efr,
            int ndims,
            int nPar,
            int nClsPar,
            int maxModes,
            int updInt,
            double Ztol,
            char root[],
            int seed,
            int *pWrap,
            bint fb,
            bint resume,
            bint outfile,
            bint initMPI,
            double logZero,
            int maxiter,
            void (*LogLike)(
                double *, int *, int *, double *, void *),
            void (*dumper)(
                int *, int *, int *, double **, double **, double **, double *,
                double *, double *, double *, void *),
            void *context,
    )


