import sys
sys.path.append(".")

import params
from util import evaluate

import os
import time
import json

from woody.util import ensure_dir_for_file
from woody.data import *          
from woody.io import DataGenerator
            
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

    else:
        raise Exception("Unknown data set!")
                        
    Xtrain, ytrain = traingen.get_all()
    Xtest, ytest = testgen.get_all() 
    
    print("")
    print("Number of training patterns:\t%i" % Xtrain.shape[0])
    print("Number of test patterns:\t%i" % Xtest.shape[0])
    print("Dimensionality of the data:\t%i\n" % Xtrain.shape[1])
    
    if param['tree_type'] == "randomized":
        from sklearn.ensemble import ExtraTreesClassifier as RF
    elif param['tree_type'] == "standard":
        from sklearn.ensemble import RandomForestClassifier as RF
    
    model = RF(
            n_estimators=param['n_estimators'],
            criterion="gini",
            max_features=param['max_features'],
            min_samples_split=2,
            n_jobs=param['n_jobs'],
            random_state=seed,
            bootstrap=param['bootstrap'],
            min_samples_leaf=1,
            max_depth=None,
            verbose=0)
    
    if profile == True:
        import yep
        assert param['n_jobs'] == 1
        yep.start("train.prof")
                    
    # training
    fit_start_time = time.time()
    model.fit(Xtrain, ytrain)
    fit_end_time = time.time()
    if profile == True:
        yep.stop()             
    ypreds_train = model.predict(Xtrain) 
     
    # testing
    test_start_time = time.time()
    ypred_test = model.predict(Xtest)
    test_end_time = time.time()
    
    results = {}
    results['dataset'] = dkey
    results['param'] = param
    results['training_time'] = fit_end_time - fit_start_time
    results['testing_time'] = test_end_time - test_start_time
    print("Training time:     %f" % results['training_time'])
    print("Testing time:      %f" % results['testing_time'])
                
    evaluate(ypreds_train, ytrain, results, "training")
    evaluate(ypred_test, ytest, results, "testing")
                    
    fname = '%s_%s_%s_%s_%s_%s.json' % (str(param['n_estimators']),
                                  str(param['max_features']),
                                  str(param['n_jobs']),
                                  str(param['bootstrap']),
                                  str(param['tree_type']),
                                  str(seed),
                                )
    fname = os.path.join(params.odir, str(dkey), str(train_size), "sk", fname)
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
