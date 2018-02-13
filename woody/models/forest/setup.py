#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import numpy

TIMING = 1

FILES_TO_BE_COMPILED_CPU = ["tree/base.c", 
                            "tree/cpu.c", 
                            "tree/tree.c", 
                            "tree/util.c", 
                            "tree/cpu/base.c", 
                            "tree/cpu/criteria.c", 
                            "tree/cpu/standard.c", 
                            "tree/cpu/fastsort.c", 
                            "timing.c", 
                            "util.c", 
                            "pqueue.c",
                       ]

DIRS_TO_BE_INCLUDED_CPU = ["tree/include", "tree/cpu/include"]

SOURCES_RELATIVE_PATH = "src/"
current_path = os.path.dirname(os.path.abspath(__file__))
sources_abs_path = os.path.abspath(os.path.join(current_path, SOURCES_RELATIVE_PATH))

# source files
source_files_cpu = [os.path.abspath(os.path.join(sources_abs_path, x)) for x in FILES_TO_BE_COMPILED_CPU]
include_paths_cpu = [os.path.abspath(os.path.join(sources_abs_path, x)) for x in DIRS_TO_BE_INCLUDED_CPU]

numpy_include = numpy.get_include()

def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration
    config = Configuration('models/forest', parent_package, top_path)

    # CPU + FLOAT
    config.add_extension("_wrapper_cpu_float", \
                                    sources = ["swig/cpu_float.i"] + source_files_cpu,
                                    swig_opts=['-modern', '-threads'],
                                    include_dirs = [numpy_include] +[include_paths_cpu],
                                    define_macros = [
                                        ('ABSOLUTE_PATH', os.path.join(sources_abs_path, "ensemble")),
                                        ('USE_DOUBLE', 0),
                                        ('TIMING', TIMING)
                                    ],
                                    libraries=['gomp'],
                                    extra_compile_args=["-std=gnu89", "-fopenmp", '-O3', '-Wall', '-pthread', '-Wno-unused-label'] + ['-I'+ipath for ipath in include_paths_cpu])

    # CPU + DOUBLE
    config.add_extension("_wrapper_cpu_double", \
                                    sources = ["swig/cpu_double.i"] + source_files_cpu,
                                    swig_opts=['-modern', '-threads'],
                                    include_dirs = [numpy_include] +[include_paths_cpu],
                                    define_macros = [
                                        ('ABSOLUTE_PATH', os.path.join(sources_abs_path, "ensemble")),
                                        ('USE_DOUBLE', 1),
                                        ('TIMING', TIMING)
                                    ],
                                    libraries=['gomp'],
                                    extra_compile_args=["-std=gnu89", "-fopenmp", '-O3', '-Wall', '-pthread', '-Wno-unused-label'] + ['-I'+ipath for ipath in include_paths_cpu])
        
    return config

if __name__ == '__main__':
        
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
