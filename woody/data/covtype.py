#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import numpy
import pandas

from woody.io import DataGenerator

from .util import check_and_download, save_to_h5pd

def get_covtype_files(data_path, train_size=100000):
    
    fname_train = os.path.join(data_path, "covtype/covtype-train-1.csv")
    fname_test = os.path.join(data_path, "covtype/covtype-test-1.csv")
    check_and_download(fname_train)
    check_and_download(fname_test)
    
    fname_train_size = os.path.join(data_path, "covtype/covtype-train-1_%s.csv" % str(train_size))
    
    if not os.path.exists(fname_train_size):
        os.system("sed -n '%i,%ip;%iq' < %s > %s" % (1, train_size, train_size, fname_train, fname_train_size))    
    
    return fname_train_size, fname_test
    
def get_covtype_data(data_path, train_size=100000, shuffle_train=False, shuffle_test=False, seed=0):

    numpy.random.seed(seed)

    fname_train, fname_test = get_covtype_files(data_path, train_size)

    # training data
    outcome_col = 55
    features = 54
    data = pandas.read_csv(fname_train, dtype="int", header=None)
    ytrain = numpy.ascontiguousarray(data[(outcome_col-1)].values)
    xcols = set(range(features+1)).difference(set([outcome_col-1]))
    Xtrain = numpy.ascontiguousarray(data.ix[:,xcols].values)
    
    if shuffle_train == True:
        train_partition = numpy.random.permutation(Xtrain.shape[0])    
        Xtrain = Xtrain[train_partition]
        ytrain = ytrain[train_partition]

    # testing data
    data = pandas.read_csv(fname_test, dtype=int, header=None)
    ytest = numpy.ascontiguousarray(data[(outcome_col-1)].values)
    xcols = set(range(features+1)).difference(set([outcome_col-1]))
    Xtest = numpy.ascontiguousarray(data.ix[:,xcols].values)
    
    if shuffle_test == True:
        test_partition = numpy.random.permutation(Xtest.shape[0])    
        Xtest = Xtest[test_partition]
        ytest = ytest[test_partition]

    return Xtrain, ytrain, Xtest, ytest

def _convert_datasets(data_path, train_size):

    X_train, y_train, X_test, y_test = get_covtype_data(data_path, train_size, shuffle_train=False, shuffle_test=False)

    fname_store_train = os.path.join(data_path, "covtype/covtype-train-1_%s.csv.h5pd" % str(train_size))
    fname_store_test = os.path.join(data_path, "covtype/covtype-test-1.csv.h5pd")

    save_to_h5pd(X_train, y_train, fname_store_train)
    save_to_h5pd(X_test, y_test, fname_store_test)

def get_covtype_generator(data_path, train_size=100000, store="h5", seed=0, part="train", patterns=True, target=True):
    

    if store == "h5":
        
        if part=="train":
            fname = os.path.join(data_path, "covtype/covtype-train-1_%s.csv.h5pd" % str(train_size))
        elif part=="test":
            fname = os.path.join(data_path, "covtype/covtype-test-1.csv.h5pd")
            
        if not os.path.exists(fname):
            print("Store for covtype data does not exist. Generating all stores ...")
            _convert_datasets(data_path, train_size)
    
        return DataGenerator(fname=fname, seed=seed, patterns=patterns, target=target, chunksize=200000)
    
    elif store == "mem":
    
        X_train, y_train, X_test, y_test = get_covtype_data(data_path, train_size=train_size, shuffle_train=False, shuffle_test=False)
        
        data = {}
        if part == "train":
            data['X'] = X_train
            data['y'] = y_train
        else:
            data['X'] = X_test
            data['y'] = y_test            
                    
        return DataGenerator(data=data, seed=seed, patterns=patterns, target=target, chunksize=200000)
    