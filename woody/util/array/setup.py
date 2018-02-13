#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import numpy

FILES_TO_BE_COMPILED_CPU = ["array.c", "util.c"]
DIRS_TO_BE_INCLUDED_CPU = ["include"]

SOURCES_RELATIVE_PATH = "src/"
current_path = os.path.dirname(os.path.abspath(__file__))
sources_abs_path = os.path.abspath(os.path.join(current_path, SOURCES_RELATIVE_PATH))

# source files
source_files_cpu = [os.path.abspath(os.path.join(sources_abs_path, x)) for x in FILES_TO_BE_COMPILED_CPU]
include_paths_cpu = [os.path.abspath(os.path.join(sources_abs_path, x)) for x in DIRS_TO_BE_INCLUDED_CPU]

numpy_include = numpy.get_include()

def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration
    config = Configuration('util/c', parent_package, top_path)

    # CPU + FLOAT
    config.add_extension("_wrapper_utils_cpu_float", \
                                    sources = ["swig/cpu_float.i"] + source_files_cpu,
                                    swig_opts=['-modern'],
                                    include_dirs = [numpy_include] +[include_paths_cpu],
                                    define_macros = [
                                        ('USE_DOUBLE', 0),
                                    ],
                                    libraries=['gomp'],
                                    extra_compile_args=["-std=gnu89", "-fopenmp", '-pthread', '-O3', '-Wall', '-Wno-unused-label'] + ['-I'+ipath for ipath in include_paths_cpu])

    # CPU + DOUBLE
    config.add_extension("_wrapper_utils_cpu_double", \
                                    sources = ["swig/cpu_double.i"] + source_files_cpu,
                                    swig_opts=['-modern'],
                                    include_dirs = [numpy_include] +[include_paths_cpu],
                                    define_macros = [
                                        ('USE_DOUBLE', 1),
                                    ],
                                    libraries=['gomp'],
                                    extra_compile_args=["-std=gnu89", "-fopenmp", '-pthread', '-O3', '-Wall', '-Wno-unused-label'] + ['-I'+ipath for ipath in include_paths_cpu])

    return config

if __name__ == '__main__':
        
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

