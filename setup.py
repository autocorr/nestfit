#/usr/bin/env python3

import os
from pathlib import Path

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

import numpy as np


MNEST_DIR = os.getenv('MNEST_DIR')
if MNEST_DIR is None:
    raise ValueError('Environment variable `MNEST_DIR` unset')
MNEST_DIR = Path(MNEST_DIR).expanduser()


ext = Extension(
        'nestfit.wrapped',
        ['nestfit/wrapper.pyx', 'nestfit/fastexp.c'],
        libraries=['m', 'multinest'],
        include_dirs=[np.get_include(), str(MNEST_DIR/'include'), 'nestfit'],
        library_dirs=[str(MNEST_DIR/'lib')],
        extra_compile_args=['-O3', '-march=native', '-mtune=native',
            '-ffast-math', '-fopenmp'],
        extra_link_args=['-fopenmp'],
        # Enable line tracing, but note performance penalty
        #define_macros=[('CYTHON_TRACE', '1')],
)


if __name__ == '__main__':
    setup(
            name='nestfit',
            version='0.1',
            author='Brian Svoboda',
            author_email='brian.e.svoboda@gmail.com',
            license='MIT',
            url='https://github.com/autocorr/nestfit',
            requires=['numpy', 'cython'],
            packages=['nestfit'],
            cmdclass={'build_ext': build_ext},
            ext_modules=[ext],
    )
