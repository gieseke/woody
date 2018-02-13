#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import shutil

from woody.io import DataGenerator

from .util import save_to_h5pd

def get_artificial_data(size=1000, seed=0):

    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=size, n_features=2, n_redundant=0, 
                               n_informative=2, random_state=seed, 
                               n_clusters_per_class=1)
    n_train = len(X) / 2
    X_train, y_train, X_test, y_test = X[:n_train], y[:n_train], X[n_train:], y[n_train:]

    return X_train, y_train, X_test, y_test

def _convert_datasets(data_path, size=1000, seed=0):

    X_train, y_train, X_test, y_test = get_artificial_data(size=size, seed=seed)
    
    fname_store_train = os.path.join(data_path, "artificial/train_" + str(size) + ".h5pd")
    fname_store_test = os.path.join(data_path, "artificial/test_" + str(size) + ".h5pd")

    save_to_h5pd(X_train, y_train, fname_store_train)
    save_to_h5pd(X_test, y_test, fname_store_test)
    
def get_artificial_generator(data_path, size=1000, seed=0, part="train", store="h5", patterns=True, target=True):

    if part=="train":
        fname = os.path.join(data_path, "artificial/train_" + str(size) + ".h5pd")
    elif part=="test":
        fname = os.path.join(data_path, "artificial/test_" + str(size) + ".h5pd")


    try:
        shutil.rmtree(fname)
    except:
        pass
    
    if not os.path.exists(fname):
        print("Store for artificial data does not exist. Generating all stores ...")            
        _convert_datasets(data_path, size=size, seed=seed)

    return DataGenerator(fname=fname, seed=seed, patterns=patterns, target=target, chunksize=200000)
