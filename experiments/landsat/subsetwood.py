import sys
sys.path.append(".")

import matplotlib.pyplot as plt 
import numpy
import os
import json
import pandas
from util import evaluate
import params

import time

from woody import SubsetWoodClassifier

from woody.io import DiskStore, DataGenerator
from woody.util import ensure_dir_for_file
from woody.data import *

label_mappings = {
1:"forest",
2:"meadow",
3:"riverbank",
4:"highway",
5:"building",
0:"background",
6:"reservoir",
7:"grassland",
8:"light_rail",
9:"farmland"
}

names = ["forest", "meadow", "riverbank", "highway", "building", "reservoir", "grassland", "rail", "farmland"]

def single_run(dkey, train_size, param, seed, profile=False):
                
    print("Processing data set %s with train_size %s and parameters %s ..." % (str(dkey), str(train_size), str(param)))
    
    tmp_dir = "tmp/subsetwood"
    
    if dkey == "landsat":

        # TODO: Download file manually if needed (9,7GB and 524MB):
        # wget https://sid.erda.dk/share_redirect/GsVMKksFSk/landsat_train_LC08_L1TP_196022_20150415_20170409_01_T1_test_random_row_0.050000.h5pd
        # wget https://sid.erda.dk/share_redirect/GsVMKksFSk/landsat_test_LC08_L1TP_196022_20150415_20170409_01_T1_test_random_row_0.050000.h5pd

        # TODO: Adapt paths accordingly
        fname_train = "data/landsat_train_LC08_L1TP_196022_20150415_20170409_01_T1_test_random_row_0.050000.h5pd"
        fname_test = "data/landsat_test_LC08_L1TP_196022_20150415_20170409_01_T1_test_random_row_0.050000.h5pd"
        
        traingen = DataGenerator(fname=fname_train, seed=seed, patterns=True, target=True, chunksize=1000000, n_lines_max=train_size)
        testgen = DataGenerator(fname=fname_test, seed=seed, patterns=True, target=True, chunksize=1000000, n_lines_max=20000000)
    
    else:
        raise Exception("Unknown data set!")

    print("")
    print("Number of training patterns:\t%i" % traingen.get_shapes()[0][0])
    print("Number of test patterns:\t%i" % testgen.get_shapes()[0][0])
    print("Dimensionality of the data:\t%i\n" % traingen.get_shapes()[0][1])
    
    # set to top trees size
    n_subset = 500000

    model = SubsetWoodClassifier(
                n_estimators=param['n_estimators'],
                criterion="gini",
                max_features=param['max_features'],
                min_samples_split=2,
                n_jobs=param['n_jobs'],
                seed=seed,
                bootstrap=param['bootstrap'],
                tree_traversal_mode="dfs",
                tree_type=param['tree_type'],
                min_samples_leaf=1,
                float_type="double",
                max_depth=None,
                verbose=1,
                odir=tmp_dir,
                store=DiskStore())

    # training
    if profile == True:
        import yep
        assert param['n_jobs'] == 1
        yep.start("train.prof")
                
    fit_start_time = time.time()        
    model.fit(traingen, n_subset=n_subset)
    fit_end_time = time.time()
    if profile == True:
        yep.stop()
    
    # testing
    print("Computing predictions ...")
    test_start_time = time.time()
    ypred_test = model.predict(generator=testgen)
    test_end_time = time.time()
    
    results = {}
    results['dataset'] = dkey
    results['param'] = param
    results['training_time'] = fit_end_time - fit_start_time
    results['testing_time'] = test_end_time - test_start_time
    results['total'] = model.get_training_times()['total']
    results['retrieve'] = model.get_training_times()['retrieve']
    results['subset'] = model.get_training_times()['subset']
    
    print("Training time:\t\t%f" % results['training_time'])
    print("Testing time:\t\t%f" % results['testing_time'])
    
    print("Evaluating test error ...")

    ytest = testgen.get_all_target()            
    ytrain = traingen.get_all_target()            
    ytrain = ytrain.astype(numpy.int64)
    ytest = ytest.astype(numpy.int64)
    ypred_test = ypred_test.astype(numpy.int64)
    evaluate(ypred_test, ytest, results, "testing")

    print("Training distribution")
    print(numpy.bincount(ytrain))

    print("Test distribution")
    print(numpy.bincount(ytest))

    print("Predict distribution")
    print(numpy.bincount(ypred_test))
    
    fname = '%s_%s_%s_%s_%s_%s.json' % (str(param['n_estimators']),
                                  str(param['max_features']),
                                  str(param['n_jobs']),
                                  str(param['bootstrap']),
                                  str(param['tree_type']),
                                  str(seed),
                                )
    fname = os.path.join(params.odir, str(dkey), str(train_size), "subsetwood_" + str(n_subset), fname)
    ensure_dir_for_file(fname)
    with open(fname, 'w') as fp:
        json.dump(results, fp)

    
    del(testgen)
    del(traingen)
    model.cleanup()
    time.sleep(1)

###################################################################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dkey', nargs='?', const="covtype", type=str, default="covtype")
parser.add_argument('--train_size', nargs='?', const=0, type=int, default=0)
parser.add_argument('--seed', nargs='?', const=0, type=int, default=0)
parser.add_argument('--key', type=str)
args = parser.parse_args()
dkey, train_size, seed, key = args.dkey, args.train_size, args.seed, args.key
###################################################################################

single_run(dkey, train_size, params.parameters[key], seed)
