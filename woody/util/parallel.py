#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import multiprocessing
from multiprocessing.pool import ThreadPool

def pool_init():  
    import gc
    gc.collect()
    
def wrapped_task(queue, task, args, kwargs):
    
    queue.put(task(*args, **kwargs))        

from multiprocessing import Queue

# https://github.com/joblib/joblib/issues/138    
def start_via_single_process(task, args, kwargs):
            
    queue = Queue()

    proc = multiprocessing.Process(target=wrapped_task, args=(queue, task, args, kwargs))                
    proc.start()
    
    result = queue.get()
    
    # joining might yield errors ...
    # https://gist.github.com/schlamar/2311116
    # see https://docs.python.org/2/library/multiprocessing.html#all-platforms
    #proc.join()
    return result
    
    
# def perform_task_in_parallel_in_place(task, params_parallel, n_jobs=1):
#     """ Performas a task in parallel (in place, not return results are generated
#      
#     Parameters
#     ----------
#     task : callable
#         The function/procedure that shall be executed
#     params_parallel : list
#         The parallel parameters
#     n_jobs : int, default 1
#         The number of jobs that shall be used
#     """
#     
#     
#     # https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing.pool
#     pool = multiprocessing.Pool(n_jobs, maxtasksperchild=1)
#     results = pool.apply_async(task, params_parallel)
# 
#     pool.close()
#     pool.join()    
#         
#     return results
    
            
def perform_task_in_parallel(task, params_parallel, n_jobs=1, backend="multiprocessing"):
    """ Performas a task in parallel
     
    Parameters
    ----------
    task : callable
        The function/procedure that shall be executed
    params_parallel : list
        The parallel parameters
    n_jobs : int, default 1
        The number of jobs that shall be used
    backend : str, default 'multiprocessing'
    """
    
    if backend == 'multiprocessing':
    
        # https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing.pool
        pool = multiprocessing.Pool(n_jobs, maxtasksperchild=1, initializer=pool_init)
        results = pool.map(task, params_parallel)
    
        pool.close()
        pool.join()    
            
        return results
    
    elif backend == 'threading':
        
        pool = ThreadPool(n_jobs)
        results = pool.map(task, params_parallel)
        pool.close()
        pool.join()

        return results            

    else:
        raise Exception("Unknown backend: %s" % str(backend))
    
    
if __name__ == "__main__":
    
    def foo(x):
        print x
        return x*x
    
    params_parallel = range(10000)
    #perform_task_in_parallel(foo, params_parallel, backend="multiprocessing", n_jobs=4)
    results = perform_task_in_parallel(foo, params_parallel, backend="multiprocessing", n_jobs=4)
    print "results=", results
    #results = perform_task_in_parallel(foo, params_parallel, backend="threading", n_jobs=4)
    #print "results=", results    
    