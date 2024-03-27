   #----------------------------------------------------------------------#
   #  distutils setup script for compiling cut-pursuit python extensions  #
   #----------------------------------------------------------------------#
""" 
Compilation command: python setup.py build_ext

Hugo Raguet 2020
"""

from setuptools import setup, Extension
from distutils.command.build import build
import numpy
import shutil # for rmtree, os.rmdir can only remove _empty_ directory
import os 
import re

###  targets and compile options  ###
name = "grid_graph"

include_dirs = [numpy.get_include(), # find the Numpy headers
                "../include"]
# compilation and linkage options
# MIN_OPS_PER_THREAD roughly controls parallelization, see doc in README.md
if os.name == 'nt': # windows
    extra_compile_args = ["/std:c++11", "/openmp",
                          "-DMIN_OPS_PER_THREAD=10000"]
    extra_link_args = ["/lgomp"]
elif os.name == 'posix': # linux
    extra_compile_args = ["-std=c++11", "-fopenmp",
                          "-DMIN_OPS_PER_THREAD=10000"]
    extra_link_args = ["-lgomp"]
else:
    raise NotImplementedError('OS not yet supported.')

###  auxiliary functions  ###
class build_class(build):
    def initialize_options(self):
        build.initialize_options(self)
        self.build_lib = "bin" 
    def run(self):
        build_path = self.build_lib

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

###  preprocessing  ###
# ensure right working directory
tmp_work_dir = os.path.realpath(os.curdir)
os.chdir(os.path.realpath(os.path.dirname(__file__)))

if not os.path.exists("bin"):
    os.mkdir("bin")

# remove previously compiled lib
purge("bin/", "grid_graph")

###  compilation  ###
mod = Extension(
        name,
        # list source files
        ["cpython/grid_graph_cpy.cpp",
         "../src/edge_list_to_forward_star.cpp",
         "../src/grid_to_graph.cpp"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)

setup(name=name, ext_modules=[mod], cmdclass=dict(build=build_class))

###  postprocessing  ###
try:
    shutil.rmtree("build") # remove temporary compilation products
except FileNotFoundError:
    pass

os.chdir(tmp_work_dir) # get back to initial working directory


# https://gitlab.com/1a7r0ch3/parallel-cut-pursuit
