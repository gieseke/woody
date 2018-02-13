#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import time
import numpy
import pandas

from woody.io import DataGenerator

from .util import check_and_download, save_to_h5pd

ALLOWED_TRAIN_SIZES = [500000, 1000000, 
                       1500000, 2000000,
                       2500000, 3000000,
                       3500000, 4000000]

def get_susy_files(data_path, train_size=1000000):

    assert train_size in ALLOWED_TRAIN_SIZES
    
    fname = os.path.join(data_path, "susy/SUSY.csv")
    check_and_download(fname)
    time.sleep(1)
        
    fname_train = os.path.join(data_path, "susy/SUSY.train_%s.csv" % str(train_size))
    fname_test = os.path.join(data_path, "susy/SUSY.test.csv")
    
    if not os.path.exists(fname_train):
        os.system("sed -n '%i,%ip;%iq' < %s > %s" % (1, train_size, train_size, fname, fname_train))
    if not os.path.exists(fname_test):
        os.system("sed -n '%i,%ip;%iq' < %s > %s" % (4500001, 5000000, 5000000, fname, fname_test))

    return fname_train, fname_test
    
def get_susy_data(data_path, train_size=1000000, shuffle_train=False, shuffle_test=False, seed=0):

    assert train_size in ALLOWED_TRAIN_SIZES
    
    numpy.random.seed(seed)
    fname_train, fname_test = get_susy_files(data_path, train_size)

    # training data
    label_col = 0
    features_cols = range(1,19)
    
    data = pandas.read_csv(fname_train, dtype="float", sep=",", header=None)    
    ytrain = numpy.ascontiguousarray(data.ix[:,label_col].values)
    Xtrain = numpy.ascontiguousarray(data.ix[:,features_cols].values)

    data = pandas.read_csv(fname_test, dtype="float", sep=",", header=None)    
    ytest = numpy.ascontiguousarray(data.ix[:,label_col].values)
    Xtest = numpy.ascontiguousarray(data.ix[:,features_cols].values)
            
    if shuffle_train == True:
        train_partition = numpy.random.permutation(Xtrain.shape[0])    
        Xtrain = Xtrain[train_partition]
        ytrain = ytrain[train_partition]

    if shuffle_test == True:
        test_partition = numpy.random.permutation(Xtest.shape[0])    
        Xtest = Xtest[test_partition]
        ytest = ytest[test_partition]

    return Xtrain, ytrain, Xtest, ytest

def _convert_susy_data(data_path, train_size):

    X_train, y_train, X_test, y_test = get_susy_data(data_path, train_size=train_size, shuffle_train=False, shuffle_test=False)

    fname_store_train = os.path.join(data_path, "susy/SUSY.train_%s.h5pd" % str(train_size))
    fname_store_test = os.path.join(data_path, "susy/SUSY.test.h5pd")

    save_to_h5pd(X_train, y_train, fname_store_train)
    save_to_h5pd(X_test, y_test, fname_store_test)

def get_susy_generator(data_path, train_size=1000000, store="h5", seed=0, part="train", patterns=True, target=True):
    
    if store == "h5":
        
        if part=="train":
            fname = os.path.join(data_path, "susy/SUSY.train_%s.h5pd" % str(train_size))
        elif part=="test":
            fname = os.path.join(data_path, "susy/SUSY.test.h5pd")
            
        if not os.path.exists(fname):
            print("Store for susy data does not exist. Generating all stores ...")
            _convert_susy_data(data_path, train_size)
    
        if part == "test":
            chunksize = 250000
        else:
            if train_size <= 2000000:
                chunksize = 500000
            else:
                chunksize = 2000000
            
        return DataGenerator(fname=fname, seed=seed, patterns=patterns, target=target, chunksize=chunksize)
    
    elif store == "mem":
    
        X_train, y_train, X_test, y_test = get_susy_data(data_path, train_size=train_size, shuffle_train=False, shuffle_test=False)
        
        data = {}
        if part == "train":
            data['X'] = X_train
            data['y'] = y_train
        else:
            data['X'] = X_test
            data['y'] = y_test            
                    
        return DataGenerator(data=data, seed=seed, patterns=patterns, target=target, chunksize=1000000)
