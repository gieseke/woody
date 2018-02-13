#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

from .base import Wood

class WoodClassifier(Wood):
    """ Random forest classifier.
    """
    
    def __init__(self,
                 seed=0,
                 n_estimators=10,
                 min_samples_split=2,
                 max_features=None,
                 bootstrap=False,
                 max_depth=None,
                 min_samples_leaf=1,
                 criterion="gini",
                 tree_traversal_mode="dfs",
                 leaf_stopping_mode="all",
                 tree_type="randomized",
                 float_type="double",
                 patts_trans=True,
                 do_patts_trans=True,
                 lam_criterion=0.0,
                 n_jobs=1,                                  
                 verbose=1,
                 ):
        
        super(WoodClassifier, self).__init__(
                 seed=seed,
                 n_estimators=n_estimators,
                 min_samples_split=min_samples_split,
                 max_features=max_features,
                 bootstrap=bootstrap,
                 max_depth=max_depth,
                 min_samples_leaf=min_samples_leaf,
                 learning_type="classification",
                 criterion=criterion,
                 tree_traversal_mode=tree_traversal_mode,
                 leaf_stopping_mode=leaf_stopping_mode,
                 tree_type=tree_type,
                 float_type=float_type,
                 patts_trans=patts_trans,
                 do_patts_trans=do_patts_trans,
                 lam_criterion=lam_criterion,
                 n_jobs=n_jobs,                                  
                 verbose=verbose)