#cython: language_level=3


cdef extern from 'math.h' nogil:
    const double M_PI
    const float NAN
    double c_abs 'abs' (double)
    double c_exp 'exp' (double)
    double c_expm1 'expm1' (double)
    double c_log 'log' (double)
    double c_sqrt 'sqrt' (double)
    double c_floor 'floor' (double)


cdef extern from 'fastexp.h' nogil:
    double fast_expn 'FastExp' (const float)
    void calcExpTableEntries(const int, const int)
    void fillErfTable()


