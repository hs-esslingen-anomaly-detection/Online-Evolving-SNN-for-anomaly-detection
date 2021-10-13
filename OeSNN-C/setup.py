# Cython compile instructions

from distutils.core import setup
from Cython.Build import cythonize

from distutils.command.build_ext import build_ext
from distutils.sysconfig import customize_compiler

# workaround for https://bugs.python.org/issue1222585
class my_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)
        
setup(
  name = "OeSSN",
  cmdclass = {'build_ext': my_build_ext},
  ext_modules = cythonize('./src/*.pyx')
)
