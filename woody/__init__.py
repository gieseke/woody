#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

""" 
The woody package aims at large-scale implementations
for random forests. It is based on an efficient C 
implementation  and resorts to distributed computing 
strategies.
"""

import warnings

try:
    from woody.models import WoodClassifier, WoodRegressor, HugeWoodClassifier, HugeWoodRegressor, SubsetWoodClassifier, SubsetWoodRegressor
except Exception as e:
    warnings.warn("Swig models not compiled yet? Error message: %s" % str(e))

__version__ = "0.4.dev0"
