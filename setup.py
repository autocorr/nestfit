#/usr/bin/env python3

import os
import sys
from glob import glob
from pathlib import Path

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

import numpy as np


ROOT = Path(os.path.abspath(os.path.dirname(__file__)))

MNEST_DIR = os.getenv('MNEST_DIR')
if MNEST_DIR is None:
    raise ValueError('Environment variable `MNEST_DIR` unset')
MNEST_DIR = Path(MNEST_DIR).expanduser()

MOD_NAMES = [
        'nestfit.core.core',
        'nestfit.models.ammonia',
        'nestfit.models.diazenylium',
        'nestfit.models.gaussian',
        'nestfit.models.hyperfine',
]


def clean(mod_name):
    filen = mod_name.replace('.', '/')
    for ext in ('so', 'html', 'cpp', 'c'):
        file_path = ROOT / f'{filen}.{ext}'
        if path.exists():
            path.remove()


def init_ext(mod_name):
    file_path = mod_name.replace('.', '/')
    ext = Extension(
            mod_name,
            [f'{file_path}.pyx', 'nestfit/core/fastexp.c'],
            libraries=['m', 'multinest'],
            include_dirs=[np.get_include(), str(MNEST_DIR/'include'),
                'nestfit/core', 'includes'],
            library_dirs=[str(MNEST_DIR/'lib')],
            extra_compile_args=['-O3', '-march=native', '-mtune=native',
                '-ffast-math', '-fopenmp'],
            extra_link_args=['-fopenmp'],
            # Enable line tracing, but note performance penalty
            #define_macros=[('CYTHON_TRACE', '1')],
    )
    ext.cython_directives = {'embedsignature': True}
    return ext


MODULES = [
        init_ext(name) for name in MOD_NAMES
]


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        for mod_name in MOD_NAMES:
            clean(mod_name)
    else:
        setup(
                name='nestfit',
                version='0.2',
                author='Brian Svoboda',
                author_email='brian.e.svoboda@gmail.com',
                license='MIT',
                url='https://github.com/autocorr/nestfit',
                requires=['numpy', 'cython'],
                packages=['nestfit'],
                cmdclass={'build_ext': build_ext},
                ext_modules=MODULES,
        )


