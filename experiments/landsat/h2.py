import sys
sys.path.append(".")

import params
from util import evaluate

import os
import pandas
import time
import json
import numpy
import math
from woody.io import DataGenerator

from woody.util import ensure_dir_for_file
from woody.data import *

def single_run(dkey, train_size, param, seed, profile=False):
            
    print("Processing data set %s with train_size %s and parameters %s ..." % (str(dkey), str(train_size), str(param)))
    
    if dkey == "landsat":

        # TODO: Download file manually if needed (9,7GB and 524MB):
        # wget https://sid.erda.dk/share_redirect/GsVMKksFSk/landsat_train_LC08_L1TP_196022_20150415_20170409_01_T1_test_random_row_0.050000.h5pd
        # wget https://sid.erda.dk/share_redirect/GsVMKksFSk/landsat_test_LC08_L1TP_196022_20150415_20170409_01_T1_test_random_row_0.050000.h5pd

        # TODO: Adapt paths accordingly
        fname_train = "data/landsat_train_LC08_L1TP_196022_20150415_20170409_01_T1_test_random_row_0.050000.h5pd"
        fname_test = "data/landsat_test_LC08_L1TP_196022_20150415_20170409_01_T1_test_random_row_0.050000.h5pd"
        
        traingen = DataGenerator(fname=fname_train, seed=seed, patterns=True, target=True, chunksize=1000000, n_lines_max=train_size)
        testgen = DataGenerator(fname=fname_test, seed=seed, patterns=True, target=True, chunksize=1000000, n_lines_max=20000000)
                
        # TODO: Adapt paths accordingly
        fname_train_csv = "tmp/landsat_train_small_%lu.csv" % train_size
        fname_test_csv = "tmp/landsat_test.csv"
        
    traingen.to_csv(fname_train_csv, cache=False, remove=True)
    testgen.to_csv(fname_test_csv, cache=False, remove=True)
            
    import h2o
    from skutil.h2o import h2o_col_to_numpy
    h2o.init(max_mem_size = "12G", nthreads=param['n_jobs']) 
    h2o.remove_all() 
    from h2o.estimators.random_forest import H2ORandomForestEstimator
    
    if dkey == "landsat_small" or dkey == "landsat":
        train_df = h2o.import_file(fname_train_csv)
        test_df = h2o.import_file(fname_test_csv)
        Xcols, ycol = train_df.col_names[:-1], train_df.col_names[-1]                
    else:
        raise Exception("Unknown data set!")
                                                
    print("")
    print("Number of training patterns:\t%i" % train_df.shape[0])
    print("Number of test patterns:\t%i" % test_df.shape[0])
    print("Dimensionality of the data:\t%i\n" % train_df.shape[1])

    if param['max_features'] is None:
        mtries = train_df.shape[1] - 2
    elif param['max_features'] == "sqrt":
        mtries = int(math.sqrt(train_df.shape[1] - 2))
    
    if param['bootstrap'] == False:
        sample_rate = 1.0
    else:
        sample_rate = 0.632
        
    model = H2ORandomForestEstimator(
                mtries=mtries,
                sample_rate=sample_rate,
                #nbins=1000, #crash
                min_rows=1,
                build_tree_one_node=True,
                max_depth=20,
                balance_classes=False,
                ntrees=param['n_estimators'],
                seed=seed)
    
    # training
    fit_start_time = time.time()
    model.train(Xcols, ycol, training_frame=train_df)
    fit_end_time = time.time()
    
    # testing
    test_start_time = time.time()
    ypreds_test = model.predict(test_df)
    test_end_time = time.time()
    
    results = {}
    results['dataset'] = dkey
    results['param'] = param
    results['training_time'] = fit_end_time - fit_start_time
    results['testing_time'] = test_end_time - test_start_time
    print("Training time:     %f" % results['training_time'])
    print("Testing time:      %f" % results['testing_time'])
    
    evaluate(numpy.rint(ypreds_test.as_data_frame().values), test_df[ycol].as_data_frame().values, results, "testing")            
                            
    fname = '%s_%s_%s_%s_%s_%s.json' % (str(param['n_estimators']),
                                  str(param['max_features']),
                                  str(param['n_jobs']),
                                  str(param['bootstrap']),
                                  str(param['tree_type']),
                                  str(seed),                                  
                                )
    fname = os.path.join(params.odir, str(dkey), str(train_size), "h2", fname)
    ensure_dir_for_file(fname)
    with open(fname, 'w') as fp:
        json.dump(results, fp)

    
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
