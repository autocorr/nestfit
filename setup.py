from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

import numpy as np


ext = Extension(
        'nestfit.wrapped',
        ['nestfit/wrapper.pyx'],
        libraries=['multinest'],
        include_dirs=[np.get_include()],
        extra_link_args=[
            '-I/users/bsvoboda/code/MultiNest/include',
            '-L/users/bsvoboda/code/MultiNest/lib',
        ],
        #define_macros=[('CYTHON_TRACE', '1')],
)


if __name__ == "__main__":
    setup(
            name='nestfit',
            version='0.0.1.dev',
            author='Brian Svoboda',
            author_email='brian.e.svoboda@gmail.com',
            license='MIT',
            url='https://github.com/autocorr/nestfit',
            requires=['numpy', 'cython'],
            packages=['nestfit'],
            cmdclass={"build_ext": build_ext},
            ext_modules=[ext],
    )
