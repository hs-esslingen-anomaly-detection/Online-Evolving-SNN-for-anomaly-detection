# Cython compile instructions

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = "OeSSN",
  ext_modules = cythonize('./src/*.pyx')
)
