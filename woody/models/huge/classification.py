#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

from woody.io import DiskStore

from .base import HugeWood

from .. import WoodClassifier

class HugeWoodClassifier(HugeWood):
    """ Large-scale contruction of a random forest on
    a single workstation (with limited memory resources).
    Each tree belonging to the ensemble is constructed 
    in a multi-stage fashion and the intermediate data
    are stored on disk (e.g., via h5py).
    """
        
    TKEY_ALL_FIT = 0
    TKEY_TOP_TREE = 1
    TKEY_DISTR_PATTS = 2
    TKEY_BOTTOM_TREES = 3
    
    MAX_RAND_INT = 10000000
    
    def __init__(self,
                 n_top="auso",
                 n_patterns_leaf="auto",
                 balanced_top_tree=True,
                 top_tree_lambda=0.0,
                 top_tree_max_depth=None,
                 top_tree_type="randomized",
                 top_tree_leaf_stopping_mode="ignore_impurity",
                 n_estimators=1,
                 n_estimators_bottom=1,
                 n_jobs=1,
                 seed=0,
                 odir=".hugewood",
                 verbose=1,           
                 plot_intermediate={},
                 chunk_max_megabytes=256,
                 wrapped_instance=WoodClassifier(),     
                 store=DiskStore(),                 
                 ):
                        
        super(HugeWoodClassifier, self).__init__(n_top=n_top,
                                                 n_patterns_leaf=n_patterns_leaf,
                                                 balanced_top_tree=balanced_top_tree,
                                                 top_tree_lambda=top_tree_lambda,
                                                 top_tree_max_depth=top_tree_max_depth,
                                                 top_tree_type=top_tree_type,
                                                 top_tree_leaf_stopping_mode=top_tree_leaf_stopping_mode,
                                                 n_estimators=n_estimators,
                                                 n_estimators_bottom=n_estimators_bottom,
                                                 n_jobs=n_jobs,
                                                 seed=seed,
                                                 odir=odir,
                                                 verbose=verbose,
                                                 plot_intermediate=plot_intermediate,
                                                 chunk_max_megabytes=chunk_max_megabytes,                                                 
                                                 wrapped_instance=wrapped_instance,
                                                 store=store)
        