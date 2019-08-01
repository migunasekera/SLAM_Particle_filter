#!/usr/bin/env python2

from distutils.core import setup
from Cython.Build import cythonize

setup(
        name = "MapUtils",
        ext_modules = cythonize("MapUtils_fclad.pyx", annotate=True)
    )

