#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os

from .covtype import get_covtype_files, get_covtype_data, get_covtype_generator
from .landsat import get_landsat_files, get_landsat_generator
from .artificial import get_artificial_data, get_artificial_generator
from .higgs import get_higgs_files, get_higgs_data, get_higgs_generator
from .susy import get_susy_files, get_susy_data, get_susy_generator

def get_data_path():

    return os.path.join(os.getcwd().split('woody')[0], "woody/data")

def artificial(train_size=1000, seed=0, data_path=None):

    if data_path is None:
        data_path = get_data_path()
        
    return get_artificial_data(size=train_size, seed=seed)

def artificial_generators(train_size=1000, seed=0, patterns=True, target=True, store="h5", data_path=None):

    if data_path is None:
        data_path = get_data_path()
    
    traingen = get_artificial_generator(data_path,
                                        size=train_size,
                                     seed=seed, 
                                     part="train", 
                                     store=store, 
                                     patterns=patterns, 
                                     target=target
                                     )
    
    testgen = get_artificial_generator(get_data_path(),
                                        size=train_size,                                       
                                    seed=seed, 
                                    part="test", 
                                    store=store, 
                                    patterns=patterns, 
                                    target=target
                                    )

    return traingen, testgen

def covtype_files(data_path=None, train_size=100000):
    
    if data_path is None:
        data_path = get_data_path()
            
    return get_covtype_files(data_path, train_size=train_size)
    
def covtype(train_size=100000, shuffle_train=False, shuffle_test=False, seed=0, data_path=None):

    if data_path is None:
        data_path = get_data_path()
        
    return get_covtype_data(data_path,
                            train_size=train_size, 
                            shuffle_train=shuffle_train, 
                            shuffle_test=shuffle_test, 
                            seed=seed
                            )

def covtype_generators(train_size=100000, seed=0, patterns=True, target=True, store="h5", data_path=None):

    if data_path is None:
        data_path = get_data_path()
        
    traingen = get_covtype_generator(data_path,
                                     train_size=train_size,
                                     seed=seed, 
                                     part="train", 
                                     store=store, 
                                     patterns=patterns, 
                                     target=target
                                     )
    
    testgen = get_covtype_generator(data_path,
                                    train_size=train_size,
                                    seed=seed, 
                                    part="test", 
                                    store=store, 
                                    patterns=patterns, 
                                    target=target
                                    )

    return traingen, testgen

def higgs_files(data_path=None, train_size=1000000):

    if data_path is None:
        data_path = get_data_path()
            
    return get_higgs_files(data_path, train_size)

def higgs(train_size=1000000, shuffle_train=False, shuffle_test=False, seed=0, data_path=None):

    if data_path is None:
        data_path = get_data_path()
        
    return get_higgs_data(data_path,
                          train_size=train_size, 
                          shuffle_train=shuffle_train, 
                          shuffle_test=shuffle_test, 
                          seed=seed
                          )

def higgs_generators(train_size=1000000, seed=0, patterns=True, target=True, store="h5", data_path=None):

    if data_path is None:
        data_path = get_data_path()
        
    traingen = get_higgs_generator(data_path,
                                   train_size=train_size,                                   
                                   seed=seed, 
                                   part="train", 
                                   store=store, 
                                   patterns=patterns, 
                                   target=target
                                   )
    
    testgen = get_higgs_generator(data_path,
                                  train_size=train_size,
                                  seed=seed, 
                                  part="test", 
                                  store=store, 
                                  patterns=patterns, 
                                  target=target
                                  )

    return traingen, testgen


def landsat_files(data_path=None, data_set="LC81950212016133LGN00", version="1_1", train_size=0):
    
    if data_path is None:
        data_path = get_data_path()
            
    return get_landsat_files(data_path, data_set=data_set, version=version, train_size=train_size)

def landsat_generators(train_size=None, data_set="LC81950212016133LGN00", version="1_1", seed=0, patterns=True, target=True, store="h5", data_path=None, chunksize=10000000):

    if data_path is None:
        data_path = get_data_path()

    traingen = get_landsat_generator(data_path,  
                                     data_set=data_set,
                                     version=version,
                                     seed=seed, 
                                     part="train", 
                                     store=store, 
                                     patterns=patterns, 
                                     target=target,
                                     chunksize=chunksize,
                                     )
    
    testgen = get_landsat_generator(data_path,
                                    data_set=data_set,
                                    version=version,                                    
                                    seed=seed, 
                                    part="test", 
                                    store=store, 
                                    patterns=patterns, 
                                    target=target,
                                    chunksize=chunksize,                                    
                                    )
    
    return traingen, testgen
    

def susy_files(data_path=None, train_size=1000000):

    if data_path is None:
        data_path = get_data_path()
            
    return get_susy_files(data_path, train_size)

def susy(train_size=1000000, shuffle_train=False, shuffle_test=False, seed=0, data_path=None):

    if data_path is None:
        data_path = get_data_path()
        
    return get_susy_data(data_path,
                          train_size=train_size, 
                          shuffle_train=shuffle_train, 
                          shuffle_test=shuffle_test, 
                          seed=seed
                          )

def susy_generators(train_size=1000000, seed=0, patterns=True, target=True, store="h5", data_path=None):

    if data_path is None:
        data_path = get_data_path()
        
    traingen = get_susy_generator(data_path,
                                   train_size=train_size,                                   
                                   seed=seed, 
                                   part="train", 
                                   store=store, 
                                   patterns=patterns, 
                                   target=target
                                   )
    
    testgen = get_susy_generator(data_path,
                                  train_size=train_size,
                                  seed=seed, 
                                  part="test", 
                                  store=store, 
                                  patterns=patterns, 
                                  target=target
                                  )

    return traingen, testgen

      