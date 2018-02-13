#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration

    config = Configuration('woody', parent_package, top_path)
    config.add_subpackage('models', subpackage_path='models')
    config.add_subpackage('models/forest', subpackage_path='models/forest')
    config.add_subpackage('tests')
    config.add_subpackage('util')
    config.add_subpackage('util/array', subpackage_path='util/array')

    return config

if __name__ == '__main__':
                
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
