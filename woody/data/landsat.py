#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
from woody.io import DataGenerator

from .util import check_and_download    

def get_landsat_files(data_path, data_set="LC81950212016133LGN00", version="1_1", train_size=0):
    
    fname_train = os.path.join(data_path, "landsat", str(data_set) + "_" + version + ".train.csv")
    fname_test = os.path.join(data_path, "landsat", str(data_set) + "_" + version + ".test.csv")
    check_and_download(fname_train)
    check_and_download(fname_test)
        
    if train_size > 0:
        fname_train_size = os.path.join(data_path, "landsat", str(data_set) + "_" + version + ".train_%i.csv" % train_size)
        if not os.path.exists(fname_train_size):
            os.system("sed -n '%i,%ip;%iq' < %s > %s" % (1, train_size, train_size, fname_train, fname_train_size))
        fname_train = fname_train_size

    return fname_train, fname_test

def get_landsat_generator(data_path, train_size=10000000, data_set="LC81950212016133LGN00", version="1_1", seed=0, part="train", store=None, patterns=True, target=True, chunksize=5000000):

    assert version in ["1_1", "3_3", "pan_1_1", "pan_3_3"]

    if part=="train":
        fname = os.path.join(data_path, "landsat", str(data_set) + "_" + version + ".train.h5pd")
    elif part=="test":
        fname = os.path.join(data_path, "landsat", str(data_set) + "_" + version + ".test.h5pd")
    check_and_download(fname)
    
    return DataGenerator(fname=fname, seed=seed, patterns=patterns, target=target, chunksize=chunksize)