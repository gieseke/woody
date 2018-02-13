#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import numpy
import multiprocessing

from woody.util import split_array

from .. import Wood                
                
def distribute_patterns(toptree, X, y, verbose=0, logger=None):

    if logger is not None:
        logger.debug("\tUsing top tree to distribute patterns to leaves ...")
            
    leaves_ids = toptree.get_leaves_ids(X)
    unique_leaves_ids, counts = numpy.unique(leaves_ids, return_counts=True)
    
    if logger is not None:
        logger.debug("\tPatterns are distributed to %i leaves of the top tree ..." % len(unique_leaves_ids))
        
    chunks = -1 * numpy.ones(int(unique_leaves_ids[-1]) + 1, dtype=numpy.int32)
    for i in xrange(len(unique_leaves_ids)):
        leaf_id = int(unique_leaves_ids[i])
        chunks[leaf_id] = i
    
    Xsubs, ysubs = {}, {}

    Xnew = split_array(X, leaves_ids, chunks, counts)
    ynew = split_array(y, leaves_ids, chunks, counts)
    
    current_count = 0
    for i in xrange(len(unique_leaves_ids)):
        leaf_id = unique_leaves_ids[i]
        cts = counts[i]
        Xsubs[leaf_id] = Xnew[current_count:current_count + cts, :]
        ysubs[leaf_id] = ynew[current_count:current_count + cts]
        current_count += cts
    
    return Xsubs, ysubs, unique_leaves_ids      

def get_XY_subsets_from_store(dset, heavy_leaf_domsize):
    
    pure = False
    
    ychunk = numpy.array(dset[:, -1])
    counts = numpy.bincount(ychunk.astype(numpy.int32))
        
    dominant = numpy.argmax(counts)
    if len(ychunk) > heavy_leaf_domsize:
        rsubset = numpy.random.choice(len(ychunk), heavy_leaf_domsize)
    else:
        rsubset = numpy.arange(len(ychunk))
    subindices = ychunk != dominant
    subindices = numpy.union1d(rsubset, subindices)
    subindices.sort()

    # random access slow in h5py, process in chunks
    Xsub, ysub = numpy.array(dset[:, :-1]), numpy.array(dset[:, -1])
    Xsub, ysub = Xsub[subindices,:], ysub[subindices]
    
    print "REMOVE UPWARDS, no XSUB needed"
    if (counts != 0).sum() == 1:
        pure = True
            
    return Xsub, ysub, pure
    
def _load_single_tree(store, fname, wrapped_instance, typ=None):
    
    assert typ in ["top", "bottom"]
    
    if typ == "top":
        return store.load(fname, Wood)
    
    elif typ == "bottom":
        return store.load(fname, wrapped_instance)
    
    return None   
