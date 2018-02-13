#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import math
import shutil
import numpy
import random
import inspect
from scipy.stats import mode

from time import gmtime, strftime
from woody.util import makedirs, Timer, perform_task_in_parallel, start_via_single_process
from woody.io import DiskStore, MemoryStore

#from .predict import predict_array
#from .fit import instantiate_single_tree_instance, fit_all_attached_bottom_trees, distribute_all_patterns

from ..base import BaseEstimator
from .. import Wood
#from woody.util.parallel import params_parallel


class SubsetWood(BaseEstimator):
    """ Simple random forest wrapper that trains
    a forest on a small subset of a (potentially very
    large training set). All random subsets are extracted
    by scanning the dataset once prior to training.
    """
    
    TKEY_ALL_FIT = 0
    TKEY_TOP_SUBSETS = 1
    TKEY_SUBSET_TREE = 2
        
    MAX_RAND_INT = 10000000

    
    def __init__(self,
                 seed=0,
                 n_estimators=10,
                 min_samples_split=2,
                 max_features=None,
                 bootstrap=True,
                 max_depth=None,
                 min_samples_leaf=1,
                 learning_type=None,
                 criterion=None,
                 tree_traversal_mode="dfs",
                 leaf_stopping_mode="all",
                 tree_type="randomized",
                 float_type="double",
                 patts_trans=True,
                 do_patts_trans=True,
                 lam_criterion = 1.0,
                 n_jobs=1,                                  
                 odir=".subsetwood",
                 verbose=1,
                 store=DiskStore(),
                 ):
        """ Instantiates a huge forest. 

        Parameters
        ----------
        
        Returns
        -------
        HugeWood :     
        """
                        
        super(SubsetWood, self).__init__(verbose=verbose,
                                       logging_name="SubsetWood",
                                       seed=seed,
                                       )
        
        self.seed = seed
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.learning_type = learning_type
        self.criterion = criterion
        self.tree_traversal_mode = tree_traversal_mode
        self.leaf_stopping_mode = leaf_stopping_mode
        self.tree_type = tree_type
        self.float_type = float_type
        self.patts_trans = patts_trans
        self.do_patts_trans = do_patts_trans
        self.lam_criterion = lam_criterion
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.odir = odir
        self.store = store
        
    def get_params(self):
        """ Returns the models's parameters
        """
        
        params = super(SubsetWood, self).get_params()
        
        params.update({"seed": self.seed, 
                "n_estimators": self.n_estimators, 
                "min_samples_split": self.min_samples_split, 
                "max_features": self.max_features, 
                "bootstrap": self.bootstrap, 
                "max_depth": self.max_depth, 
                "min_samples_leaf": self.min_samples_leaf, 
                "learning_type": self.learning_type, 
                "criterion": self.criterion,
                "leaf_stopping_mode": self.leaf_stopping_mode, 
                "tree_traversal_mode": self.tree_traversal_mode, 
                "tree_type": self.tree_type, 
                "float_type": self.float_type, 
                "patts_trans": self.patts_trans, 
                "do_patts_trans": self.do_patts_trans,
                "lam_criterion": self.lam_criterion, 
                "n_jobs": self.n_jobs, 
                "verbose": self.verbose,
                "odir": self.odir, 
                "store": self.store,
                })
                
        return params
                            
    def fit(self, generator, n_subset=None):
        """ Fits the huge wood.
        
        Parameters
        ----------
        generator : DataGenerator
            A data generator instance that 
            is used to iterate over very 
            huge data files.
            
        Returns
        -------
        HugeWood : The fitted huge 
            wood instance (self)
        """
                        
        # create output directory and call super fit
        self.odir = os.path.join(self.odir, strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        makedirs(self.odir)
        super(SubsetWood, self).fit(logging_file=os.path.join(self.odir, "logger.log"))        
        
        # set numpy float and int dtypes
        if self.float_type == "float":
            self._numpy_dtype_float = numpy.float32
        else:
            self._numpy_dtype_float = numpy.float64
        self._numpy_dtype_int = numpy.int32
                
        generator.reset()
        Xshape, _ = generator.get_shapes()
        
        if n_subset is None:
            n_subset = Xshape[0]
        if n_subset > Xshape[0]:
            n_subset = Xshape[0]         
                        
        # random generator and timers
        self._randomgen = random.Random(self.seed)
        self._timers = {i: Timer() for i in range(10)}
        
        self._logger.debug("Fitting forest ...")
        generator.set_seed(self.seed)
                        
        self._timers[self.TKEY_ALL_FIT].start()
        
        self._timers[self.TKEY_TOP_SUBSETS].start()
        self._logger.debug("(I) Retrieving random subsets for random subset trees ...")
        self._retrieve_subsets(generator, n_subset)
        self._timers[self.TKEY_TOP_SUBSETS].stop()
        
        self._timers[self.TKEY_SUBSET_TREE].start()
        self._logger.debug("(II) Fitting all subset trees ...")
        self._build_subset_trees(generator)
        self._timers[self.TKEY_SUBSET_TREE].stop()
                
        self._timers[self.TKEY_ALL_FIT].stop()
        
        self._log_fitting_stats()
        
        return self
    
    def predict(self, X=None, generator=None):
        """ Computes prediction for new patterns.
        
        Parameters
        ----------
        X : array-like, default None
            A numpy array containing the patterns,
            one row per pattern. If X is not None, 
            the predictions are computed via X, 
            otherwise based on the generator.
        generator : DataGenerator, default None
            A data generator instance that 
            is used to iterate over the patterns
            
        Returns
        -------
        array-like : Predictions, one for each
            instance given in X or generator.
        """ 
               
        if X is not None:
            return self._predict_array(X)
        else:
            return self._predict_generator(generator)    
    
    def get_training_times(self):

        times = {}
        
        times['total'] = self._timers[self.TKEY_ALL_FIT].get_elapsed_time()
        times['retrieve'] = self._timers[self.TKEY_TOP_SUBSETS].get_elapsed_time()
        times['subset'] = self._timers[self.TKEY_SUBSET_TREE].get_elapsed_time()
        
        return times
    
    def cleanup(self):
        
        try:
            shutil.rmtree(self.odir, ignore_errors=True)
        except:
            pass
         
    def _retrieve_subsets(self, generator, n_subset):
        """
        """
        
        self._logger.debug("Retrieving random subsets for all estimators...")
        seed = self._randomgen.randint(0, self.MAX_RAND_INT)
        
        subsets = generator.get_multiple_random_subsets(self.n_estimators, n_subset, seed=seed)
        
        self._logger.debug("Storing subsets for all estimators ...")        
        for b in xrange(self.n_estimators):
            cname = os.path.join(self.odir, str(b), "sub.h5")
            self._logger.debug("Storing subset %s ..." % cname)
            Xtop, ytop = subsets[b]
            ytop = ytop.reshape((len(ytop), 1))
            self.store.create_dataset(cname, "X", Xtop)
            self.store.create_dataset(cname, "y", ytop)
        
    def _build_subset_trees(self, generator):
        """ Build trees for small, random subsets 
        of the data (in-place)
        """
        
        params_model = self.get_params()
        
        params_parallel = []
        
        for b in xrange(self.n_estimators):

            self._logger.debug("Fitting top tree for estimator %i ..." % b)
            
            params = [generator, self.store, self.odir, params_model, b]
            params_parallel.append(params)
            
        perform_task_in_parallel(_fit_subset_tree, params_parallel, n_jobs=self.n_jobs, backend="threading")
      
    def _predict_generator(self, generator):
 
        preds = []
        
        # reset generator, only patterns are needed
        generator.set_mode(patterns=True, target=False)
        generator.reset()
 
        while True:
            
            try:
                X_chunk = generator.get_chunk()
                X_chunk = self._ensure_dtype(X_chunk)
                assert X_chunk.shape[0] > 0    
            except:
                break
            
            self._logger.info("Processing chunk of size %i ..." % X_chunk.shape[0])
                        
            #preds_chunk = start_via_single_process(predict_array, args, {})
            preds_chunk = self._predict_array(X_chunk)
            
            # FIXME: This might eventually also lead to memory problems ...  -> append to store as well.
            preds = numpy.concatenate([preds, preds_chunk], axis=0)
                
        return preds
    
    def _log_fitting_stats(self):
        
        total = self._timers[self.TKEY_ALL_FIT].get_elapsed_time()
        percent_retrieve = 100 * (self._timers[self.TKEY_TOP_SUBSETS].get_elapsed_time()) / total
        percent_top = 100 * (self._timers[self.TKEY_SUBSET_TREE].get_elapsed_time()) / total
        
        self._logger.debug("------------------------------------------------------------------------------")
        self._logger.debug("Fitting Statistics")
        self._logger.debug("------------------------------------------------------------------------------")
        self._logger.debug("(I)\tRetrieving subsets: \t\t\t%.3f (s)\t[%2.2f %%]" % (self._timers[self.TKEY_TOP_SUBSETS].get_elapsed_time(), percent_retrieve))
        self._logger.debug("(II)\tTop tree constructions: \t\t%.3f (s)\t[%2.2f %%]" % (self._timers[self.TKEY_SUBSET_TREE].get_elapsed_time(), percent_top))
        self._logger.debug("\t\t\t\t\t\t\t%.3f (s)\t[%2.2f %%]" % (self._timers[self.TKEY_ALL_FIT].get_elapsed_time(), 100))
        self._logger.debug("------------------------------------------------------------------------------")    
    
    def _get_next_chunk(self, generator):
        """
        """
        
        X_chunk, y_chunk = generator.get_chunk()
        X_chunk, y_chunk = self._ensure_dtype(X_chunk), self._ensure_dtype(y_chunk)
        
        return X_chunk, y_chunk
    
    def _ensure_dtype(self, a):
        """
        """
        
        if a.dtype != self._numpy_dtype_float:
            a = a.astype(self._numpy_dtype_float)
            
        return a      

    def _predict_array(self, X):
        
        results = []
        for b in xrange(self.n_estimators):
            fname = os.path.join(self.odir, str(int(b)) + ".tree")
            tree = self.store.load(fname, Wood)
            preds = tree.predict(X)
            results.append(preds)
            
                    
        allpreds = numpy.zeros((len(X), self.n_estimators), dtype=self._numpy_dtype_float)
        for i in xrange(len(results)):
            allpreds[:,i] = results[i] 
        allpreds = numpy.array(allpreds)
        
        preds = self._combine_preds(allpreds)
        
        return preds 
    
    def _combine_preds(self, allpreds):
        
        if self.learning_type == "regression":
            
            preds = allpreds.mean(axis=1)
            
        elif self.learning_type == "classification":
            
            preds, _ = mode(allpreds, axis=1)
            preds = preds[:, 0]
            
        else:
            raise Exception("Unknown learning type for wrapped instance: %s" % self.learning_type)
        
        if preds.dtype != self._numpy_dtype_float:
            preds = preds.astype(self._numpy_dtype_float)
            
        return preds       
    
def _fit_subset_tree(args):
    """
    """
    
    generator, store, odir, params, estimator_id = args

    #self._logger.debug("Loading random subset ...")
    cname = os.path.join(odir, str(estimator_id), "sub.h5")
    Xtop = store.get_dataset(cname, "X")
    ytop = store.get_dataset(cname, "y")
    ytop = ytop.reshape(ytop.shape[0])
    
    #self._logger.debug("Fitting subset tree ...")
    #params = self.get_params()
    wood_params = inspect.getargspec(Wood.__init__).args
    wood_params.remove("self")
    model_params = {}
    for k in wood_params:
        model_params[k] = params[k]
    
    model_params['n_estimators'] = 1
    model_params['seed'] = estimator_id
    tree = Wood(**model_params)
    tree.fit(Xtop, ytop)

    #self._logger.debug("Saving top tree for estimator %i ..." % b)
    store.save(os.path.join(odir, str(int(estimator_id)) + ".tree"), tree)
    