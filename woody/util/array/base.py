#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import numpy

import wrapper_utils_cpu_float, wrapper_utils_cpu_double

def split_array_chunk(a, indicator, chunks, counts):
    
    if type(a[0,0]) == numpy.float64:
        wrapper = wrapper_utils_cpu_double
    elif type(a[0,0]) == numpy.float32:
        wrapper = wrapper_utils_cpu_float
    else:
        raise Exception("Invalid dtype for array: %s" % str(type(a[0,0])))
            
    anew = numpy.empty(a.shape, dtype=a.dtype)
                
    cumsums = numpy.cumsum(counts).astype(numpy.int32)
        
    wrapper.split_array(a, anew, indicator, chunks, cumsums)
        
    return anew
           
def split_array(a, indicator, chunks, counts, n_jobs=1):
    """ Splits an array according to an indicator array.
    
    Parameters
    ----------
    a : array, numpy-like
        The input array that is supposed to
        be split according to the indicator array.
    indicator: array, numpy-like
        The array that contains the indices 
        according to which the array should be 
        split up. Each index is also contained in
        the chunks array (see below).
    chunks: array, numpy-like
        This array contains all possible chunk indices
        that occur in the indices array. E.g., 
        chunks = [-1,-1,0,-1,-1,1] means that we
        have two chunks in total and an indicator 
        index 2 is mapped to chunk 0 and an 
        indicator index 5 to chunk 1.  
    counts: array, numpy-like
    """
    
    reshaped = False
        
    if len(a.shape) == 1:
        reshaped = True
        a = a.reshape((len(a), 1))

    if type(a[0,0]) == numpy.float64:
        wrapper = wrapper_utils_cpu_double
    elif type(a[0,0]) == numpy.float32:
        wrapper = wrapper_utils_cpu_float
    else:
        raise Exception("Invalid dtype for array: %s" % str(type(a[0,0])))
    
    indicator = indicator.astype(numpy.int32)
    chunks = chunks.astype(numpy.int32)
    counts = counts.astype(numpy.int32)
    
#     sanity_check = True
#     # sanity checks (to be removed)    
#     if sanity_check:
#         for indi in indicator:
#             assert chunks[indi] != -1        
#          
#         anew_check = numpy.empty(a.shape, dtype=a.dtype)
#         # compute splits
#         counter = 0
#         unique, unique_counts = numpy.unique(indicator, return_counts=True)
#         for i in xrange(len(unique)):
#             u = unique[i]
#             selector = indicator == u
#              
#             sub = a[selector,:]
#             anew_check[counter:counter+len(sub),:] = sub
#             counter += len(sub)
        
    # compute new array        
    anew = numpy.empty(a.shape, dtype=a.dtype)
    cumsums = numpy.cumsum(counts).astype(numpy.int32)
    cumsums_minus_counuts = cumsums - counts
    wrapper.split_array(a, anew, indicator, chunks, cumsums_minus_counuts)
    #anew = anew_check

#     if sanity_check == True:
#         assert numpy.allclose(anew_check, anew)
    
    if reshaped == True:
        a = a.reshape(a.shape[0])
        anew = anew.reshape(anew.shape[0])
        
    return anew

def transpose_array(a, a_trans):

    if type(a[0,0]) == numpy.float64:
        wrapper = wrapper_utils_cpu_double
    else:
        wrapper = wrapper_utils_cpu_float

    wrapper.transpose_array(a, a_trans)            
    
