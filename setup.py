#
# Copyright (C) 2015-2018 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v3
#

import os
import sys
import shutil
from distutils.command.clean import clean
 
DISTNAME = 'woody'
DESCRIPTION = 'A Python library for large-scale random forests.'
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Fabian Gieseke'
MAINTAINER_EMAIL = 'fabian.gieseke@di.ku.dk'
URL = 'https://github.com/gieseke/woody'
LICENSE = 'GNU GENERAL PUBLIC LICENSE Version 3'
DOWNLOAD_URL = 'https://github.com/gieseke/woody'

import woody
VERSION = woody.__version__

# adapted from scikit-learn
if len(set(('develop', 'release')).intersection(sys.argv)) > 0:
    import setuptools
    extra_setuptools_args = dict(zip_safe=False)
else:
    extra_setuptools_args = dict()
    
def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    config.add_subpackage('woody')

    return config

class CleanCommand(clean):
    
    description = "Cleaning up code ..."

    def run(self):

        clean.run(self)

        # remove hidden '~' files
        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                if filename.endswith('~') or filename.endswith('.pyc'):
                    os.unlink(os.path.join(dirpath, filename))

        # build related files and directories
        if os.path.exists('build'):
            shutil.rmtree('build')
        if os.path.exists('woody.egg-info'):
            shutil.rmtree('woody.egg-info')
        if os.path.exists('docs/_build'):
            shutil.rmtree('docs/_build')

        # remaining files and directories in woody dir (recursively)
        for dirpath, dirnames, filenames in os.walk('woody'):
            
            for filename in filenames:
                if (filename.endswith('.so') or 
                    filename.endswith('.pyd') or 
                    filename.endswith('.dll') or 
                    filename.endswith('.pyc') or 
                    filename.endswith('_wrap.c') or 
                    filename.startswith('wrapper_') or 
                    filename.endswith('~')):
                        os.unlink(os.path.join(dirpath, filename))

            for dirname in dirnames:
                if dirname == '__pycache__' or dirname == 'build' or dirname == '_build':
                    shutil.rmtree(os.path.join(dirpath, dirname))

        try:
            shutil.rmtree("dist")
        except:
            pass

def setup_package():
    
    metadata = dict(name=DISTNAME,
                    maintainer=MAINTAINER,
                    maintainer_email=MAINTAINER_EMAIL,
                    description=DESCRIPTION,
                    license=LICENSE,
                    url=URL,
                    version=VERSION,
                    download_url=DOWNLOAD_URL,
                    long_description=LONG_DESCRIPTION,
                    classifiers=[
                                 'Intended Audience :: Science/Research',
                                 'Intended Audience :: Developers',
                                 'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
                                 'Programming Language :: C',
                                 'Programming Language :: Python',
                                 'Programming Language :: Python :: 2',
                                 'Programming Language :: Python :: 2.6',
                                 'Programming Language :: Python :: 2.7',
                                 ],
                    cmdclass={'clean': CleanCommand},
                    install_requires=["numpy>=1.6.1"],
                    include_package_data=True,
                    package_data={'woody': []},
                    **extra_setuptools_args)

    if (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or sys.argv[1] in ('--version', 'clean'))):

        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup
        metadata['version'] = VERSION

    else:

        try:
            from numpy.distutils.core import setup
            metadata['configuration'] = configuration
        except:
            print("woody requires numpy>=1.6.1")
            sys.exit(0)

    setup(**metadata)

if __name__ == "__main__":
    
    setup_package()
    
