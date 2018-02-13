#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import pandas

def train_test_split_csv(fname, fname_train, fname_test, train_size=None, test_size=None, chunksize=500000):
    
    pandas.read_csv(fname, iterator=True, chunksize=chunksize)

def train_test_split_h5pd(fname, fname_train, fname_test, train_size=None, test_size=None):
    
    pass