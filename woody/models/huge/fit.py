#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import copy
import numpy

from woody.util import perform_task_in_parallel, start_via_single_process
from woody.io import DiskStore
from .util import distribute_patterns

from .. import WoodClassifier, WoodRegressor     

def fit_all_attached_bottom_trees(odir, params, store, logger, seed, n_jobs):
    """
    """
    
    wrapped_instance = params['wrapped_instance']
    plot_intermediate = params['plot_intermediate']
    n_estimators_bottom = params['n_estimators_bottom']
    
    fname_store = os.path.join(odir, "data.h5")
    unique_leaf_ids = store.get_keys(fname_store)
    
    n_instances_total = 0
    
    params_parallel = []
    
    verbose = 0
    if logger is not None:
        verbose = 1
        
    for i in xrange(len(unique_leaf_ids)):
        
        leaf_id = unique_leaf_ids[i]
        
        args = [wrapped_instance, fname_store, store, odir, leaf_id, seed, i, unique_leaf_ids, n_estimators_bottom, wrapped_instance.min_samples_split, None, plot_intermediate, verbose]        
        params_parallel.append(args)
    
    #results = perform_task_in_parallel(_fit_single_bottom_tree, params_parallel, n_jobs=n_jobs, backend="multiprocessing")
    
    results = []
    for param in params_parallel:
        if type(store) == DiskStore:
            res = start_via_single_process(_fit_single_bottom_tree, param, {})
        else:
            res = _fit_single_bottom_tree(*param)
        results.append(res)
    
    for res in results:
        n_instances_total += res
           
    if logger is not None:   
        logger.debug("Fitted bottom trees for %i instances in total." % n_instances_total)

def _fit_single_bottom_tree(wrapped_instance, fname_store, store, odir, leaf_id, seed, i, unique_leaf_ids, n_estimators_bottom, min_samples_split, logger, plot_intermediate, verbose):
    
    # = args
    
    dset = store.get_dataset(fname_store, leaf_id)     
    Xsub, ysub = dset[:, :-1], dset[:, -1]
    Xsub = numpy.ascontiguousarray(Xsub)
    ysub = numpy.ascontiguousarray(ysub)    
    
    n_instances = len(ysub)
    if verbose > 0:
        print("\t[%i/%i] Fitting bottom subforest %s for %i patterns ..." % (i + 1, len(unique_leaf_ids), leaf_id, len(ysub)))

    btree = instantiate_single_tree_instance(wrapped_instance, 
                                                       seed,
                                                       n_estimators=n_estimators_bottom,
                                                       min_samples_split=min_samples_split,
                                                       typ="bottom")
    btree.fit(Xsub, ysub)
            
    for j in xrange(n_estimators_bottom):                                                
        n_nodes = btree.get_n_nodes(j)
        if verbose > 1:
            print("\t -> Fitted single bottom tree %i of leaf %s has %i nodes." % (j, leaf_id, n_nodes))            
    store.save(os.path.join(odir, leaf_id + ".tree"), btree)
    
    if "bottom" in plot_intermediate.keys():
        for j in xrange(n_estimators_bottom):
            if verbose > 1:
                print("\t -> Plotting single tree %i ..." % (j))                    
            btree.draw_tree(j, fname=os.path.join(odir, leaf_id, str(j) + ".pdf"), **plot_intermediate['bottom'])            
        
    return n_instances

def instantiate_single_tree_instance(wrapped_instance,
                                      seed, 
                                      n_estimators=1, 
                                      top_tree_max_depth=None,
                                      balanced_top_tree=True, 
                                      top_tree_lambda=0.0,
                                      top_tree_type="standard", 
                                      top_tree_leaf_stopping_mode="all",
                                      min_samples_split=2, 
                                      typ=None):
    
    assert typ in ["top", "bottom"]
    
    n_jobs = 1 #wrapped_instance.n_jobs
    
    if typ == "top":
        
        if wrapped_instance.learning_type == "classification":
            
            tree = WoodClassifier(n_estimators=n_estimators,
                        criterion=wrapped_instance.criterion,
                        max_features=wrapped_instance.max_features,
                        min_samples_split=min_samples_split,                         
                        n_jobs=n_jobs,
                        seed=seed,
                        # no bootstrapping for the top trees
                        bootstrap=False,
                        min_samples_leaf=wrapped_instance.min_samples_leaf,
                        tree_traversal_mode=wrapped_instance.tree_traversal_mode,
                        leaf_stopping_mode=top_tree_leaf_stopping_mode,
                        tree_type=top_tree_type,
                        float_type=wrapped_instance.float_type,
                        max_depth=wrapped_instance.max_depth,
                        verbose=0)
            
            if top_tree_max_depth is not None:
                tree.max_depth = top_tree_max_depth
            
            if balanced_top_tree == True:
                tree.criterion = "even_" + tree.criterion
                tree.max_features = None  
            tree.lam_criterion = top_tree_lambda                  
                            
        elif wrapped_instance.learning_type == "regression":
            
            tree = WoodRegressor(n_estimators=n_estimators,
                        criterion=wrapped_instance.criterion,
                        max_features=wrapped_instance.max_features,
                        min_samples_split=min_samples_split,
                        n_jobs=n_jobs,
                        seed=seed,
                        # no bootstrapping for the top trees
                        bootstrap=False,
                        min_samples_leaf=wrapped_instance.min_samples_leaf,
                        tree_traversal_mode=wrapped_instance.tree_traversal_mode,
                        leaf_stopping_mode=top_tree_leaf_stopping_mode,
                        tree_type=top_tree_type,
                        float_type=wrapped_instance.float_type,
                        max_depth=wrapped_instance.max_depth,
                        verbose=0)       
                     
            if top_tree_max_depth is not None:
                tree.max_depth = top_tree_max_depth
            
            if balanced_top_tree == True:
                tree.criterion = "even_" + tree.criterion
                tree.max_features = None    
            tree.lam_criterion = top_tree_lambda

                     
        else:
            
            raise Exception("Invalid learning_type for wrapped instance: %s" % str(wrapped_instance.learning_type))
                
    elif typ == "bottom":
        
        tree = copy.deepcopy(wrapped_instance)
        
        tree.n_estimators = n_estimators
        tree.min_samples_split = min_samples_split
        tree.n_jobs = wrapped_instance.n_jobs
                
        # important: use new seed!
        tree.seed = seed
        tree.verbose = 0
        
        # keep bootstrap parameter!                           
                        
    return tree  

def distribute_all_patterns(args):
    """
    """
    
    X_chunk, y_chunk, toptree, odir, store, stats, logger = args
    fname_store = os.path.join(odir, "data.h5")
                
    Xsubs, ysubs, unique_leaves_ids = distribute_patterns(toptree, X_chunk, y_chunk, logger=logger)

    if logger is not None:
        logger.debug("\tStoring all leaf buckets to store ...")
                    
    # store arrays to appropriate dataset
    for leaf_id in unique_leaves_ids:
        
        Xsub, ysub = Xsubs[leaf_id], ysubs[leaf_id]
        ysub = ysub.reshape((len(ysub), 1))
        data = numpy.concatenate((Xsub, ysub), axis=1)
        data_key = str(int(leaf_id))
        
        store.append_to_dataset(fname_store, data_key, data)            
        
        if stats is not None:
            if leaf_id not in stats.keys():
                stats[int(leaf_id)] = 0
            stats[int(leaf_id)] += len(ysubs[leaf_id])
    