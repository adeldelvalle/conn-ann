"""Builds the Cython extension.  Metadata lives in pyproject.toml.

    pip install -e .            # normal install
    python setup.py build_ext --inplace   # in-place, for development
"""
from setuptools import setup, Extension

import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:  # building from a released sdist that ships the generated C
    cythonize = None

SOURCE = "conn_ann/fast/_vote.pyx" if cythonize else "conn_ann/fast/_vote.c"

ext = [Extension(
    "conn_ann.fast._vote",
    [SOURCE],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-funroll-loops"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)]

setup(ext_modules=cythonize(ext, compiler_directives={"language_level": "3"}) if cythonize else ext)
