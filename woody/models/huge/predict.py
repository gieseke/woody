
import os
import gc
import numpy
from scipy.stats import mode

from woody.io import DiskStore
from woody.util import perform_task_in_parallel
from .util import distribute_patterns
from .util import _load_single_tree
        
def predict_array(X, n_estimators, n_estimators_bottom, numpy_dtype_float, odir, store, wrapped_instance, n_jobs):
    """ Returns predictions for a given set of patterns.
    """
    
    params_parallel = []
    
    for b in xrange(n_estimators):

        odir_local = os.path.join(odir, str(int(b)))
        fname = os.path.join(odir_local, "toptree.tree")
        toptree = _load_single_tree(store, fname, wrapped_instance, typ="top")
        args = [n_estimators_bottom, toptree, X, odir_local, store, wrapped_instance, numpy_dtype_float]
        params_parallel.append(args)
    
    if type(store) == DiskStore:
        results = perform_task_in_parallel(predict_bottom, params_parallel, n_jobs=n_jobs, backend="multiprocessing")
    else:
        results = []
        for param in params_parallel:
            res = predict_bottom(param)
            results.append(res)    
    allpreds = numpy.zeros((len(X), n_estimators*n_estimators_bottom), dtype=numpy_dtype_float)
    for i in xrange(len(results)):
        allpreds[:,i*n_estimators_bottom:(i+1)*n_estimators_bottom] = results[i] 
    allpreds = numpy.array(allpreds)
    
    preds = _combine_preds(allpreds, wrapped_instance.learning_type, numpy_dtype_float)
    
    return preds  

def predict_bottom(args):
    """ FIXME: This is by far the slowest part during prediction.
    """
    
    n_estimators_bottom, toptree, X, odir, store, wrapped_instance, numpy_dtype_float = args
    
    preds = numpy.zeros((len(X), n_estimators_bottom), dtype=numpy_dtype_float)
    
    oindices = numpy.array(xrange(len(X)), dtype=numpy.float64)

    Xsubs, isubs, unique_leaves_ids = distribute_patterns(toptree, X, oindices)

    for leaf_id in unique_leaves_ids:
        isubs[leaf_id] = isubs[leaf_id].astype(numpy.int64)
    unique_leaves_ids = unique_leaves_ids.astype(numpy.int64)
    
    for leaf_id in unique_leaves_ids:
        fname = os.path.join(odir, str(int(leaf_id)) + ".tree")            
        btree = _load_single_tree(store, fname, wrapped_instance, typ="bottom")
        pleaf = btree.predict_all(Xsubs[leaf_id])
        preds[isubs[leaf_id], :] = pleaf
        
        del btree
        gc.collect()
    
    return preds

def _combine_preds(allpreds, learning_type, numpy_dtype_float):
    
    if learning_type == "regression":
        
        preds = allpreds.mean(axis=1)
        
    elif learning_type == "classification":
        
        preds, _ = mode(allpreds, axis=1)
        preds = preds[:, 0]
        
    else:
        raise Exception("Unknown learning type for wrapped instance: %s" % learning_type)
    
    if preds.dtype != numpy_dtype_float:
        preds = preds.astype(numpy_dtype_float)
        
    return preds  
    