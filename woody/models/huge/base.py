#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import math
import shutil
import numpy
import random

from time import gmtime, strftime
from woody.util import makedirs, Timer, perform_task_in_parallel, start_via_single_process
from woody.io import DiskStore, MemoryStore

from .predict import predict_array
from .fit import instantiate_single_tree_instance, fit_all_attached_bottom_trees, distribute_all_patterns

from ..base import BaseEstimator
from .. import Wood

class HugeWood(BaseEstimator):
    """ Large-scale contruction of a random forest on
    a single workstation (with limited memory resources).
    Each tree belonging to the ensemble is constructed 
    in a multi-stage fashion and the intermediate data
    are stored on disk (e.g., via h5py).
    """
        
    TKEY_ALL_FIT = 0
    TKEY_TOP_SUBSETS = 1
    TKEY_TOP_TREE = 2
    TKEY_DISTR_PATTS = 3
    TKEY_SANITY_CHECK_LEAVES = 4
    TKEY_BOTTOM_TREES = 5
    
    MAX_RAND_INT = 10000000
    
    def __init__(self,
                 n_top="auto",
                 n_patterns_leaf="auto",
                 balanced_top_tree=True,
                 top_tree_lambda=0.0,
                 top_tree_max_depth=None,
                 top_tree_type="standard",
                 top_tree_leaf_stopping_mode="ignore_impurity",
                 n_estimators=1,
                 n_estimators_bottom=1,
                 n_jobs=1,
                 seed=0,
                 odir=".hugewood",
                 verbose=1,
                 plot_intermediate={},
                 chunk_max_megabytes=256,
                 heavy_leaf_domsize=10000,                            
                 wrapped_instance=None,
                 store=DiskStore(),
                 ):
        """ Instantiates a huge forest. 

        Parameters
        ----------
        n_top : str or int, default 'auto'
            Number of instances for the top trees. If set 
            to 'auto', then a reasonable value is chosen 
            automatically. Otherwise, the number specified
            will be used. 
        
        Returns
        -------
        HugeWood :     
        """
                        
        super(HugeWood, self).__init__(verbose=verbose,
                                       logging_name="HugeWood",
                                       seed=seed,
                                       )
        self.n_top = n_top
        self.n_patterns_leaf = n_patterns_leaf
        self.balanced_top_tree = balanced_top_tree     
        self.top_tree_lambda = top_tree_lambda
        self.top_tree_max_depth = top_tree_max_depth
        self.top_tree_type = top_tree_type
        self.top_tree_leaf_stopping_mode = top_tree_leaf_stopping_mode                   
        self.n_estimators = n_estimators
        self.n_estimators_bottom = n_estimators_bottom
        self.n_jobs = n_jobs        
        self.odir = odir
        self.verbose = verbose
        self.plot_intermediate = plot_intermediate
        self.chunk_max_megabytes = chunk_max_megabytes
        self.heavy_leaf_domsize = heavy_leaf_domsize
        self.wrapped_instance = wrapped_instance
        self.store = store
        
    def get_params(self):
        """ Returns the models's parameters
        """
        
        params = super(HugeWood, self).get_params()
        
        params.update({"n_top": self.n_top,
                       "n_patterns_leaf": self.n_patterns_leaf,
                       "balanced_top_tree": self.balanced_top_tree,
                       "top_tree_lambda": self.top_tree_lambda,
                       "top_tree_max_depth": self.top_tree_max_depth,
                       "top_tree_type": self.top_tree_type,
                       "top_tree_leaf_stopping_mode": self.top_tree_leaf_stopping_mode, 
                       "n_estimators": self.n_estimators,
                       "n_estimators_bottom": self.n_estimators_bottom,
                       "n_jobs": self.n_jobs,
                       "odir": self.odir,
                       "verbose": self.verbose,
                       "plot_intermediate": self.plot_intermediate,
                       "chunk_max_megabytes": self.chunk_max_megabytes,     
                       "heavy_leaf_domsize": self.heavy_leaf_domsize,                  
                       "wrapped_instance": self.wrapped_instance,
                       "store": self.store,
                       })
        
        return params
                            
    def fit(self, generator):
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
        super(HugeWood, self).fit(logging_file=os.path.join(self.odir, "logger.log"))        

        # set numpy float and int dtypes
        if self.wrapped_instance.float_type == "float":
            self._numpy_dtype_float = numpy.float32
        else:
            self._numpy_dtype_float = numpy.float64
        self._numpy_dtype_int = numpy.int32
        
        generator.reset()
                        
        # random generator and timers
        self._randomgen = random.Random(self.seed)
        self._timers = {i: Timer() for i in range(10)}
        
        self._logger.debug("Fitting forest ...")
        generator.set_seed(self.seed)
        
        Xshape, _ = generator.get_shapes()
        
        if self.n_top == "auto":
            self.n_top = int(100 * math.sqrt(Xshape[0]))
            self.n_top = max(self.n_top, 100000)
            self.n_top = min(self.n_top, 500000)
            self.n_top = min(self.n_top, Xshape[0])
            self._logger.info("Setting n_top to %s." % str(self.n_top))
            
        if self.n_patterns_leaf == "auto":
            self.n_patterns_leaf = int(100 * math.sqrt(Xshape[0]))
            self.n_patterns_leaf = max(self.n_patterns_leaf, 100000)
            self.n_patterns_leaf = min(self.n_patterns_leaf, 500000)
            self.n_patterns_leaf = min(self.n_patterns_leaf, Xshape[0] / 2)
            self._logger.info("Setting n_patterns_leaf to %s." % str(self.n_patterns_leaf))
        
        self._timers[self.TKEY_ALL_FIT].start()
        
        self._timers[self.TKEY_TOP_SUBSETS].start()
        self._logger.debug("(I) Retrieving random subsets for top trees ...")
        self._retrieve_top_subsets(generator)
        self._timers[self.TKEY_TOP_SUBSETS].stop()
        
        self._timers[self.TKEY_TOP_TREE].start()
        self._logger.debug("(II) Fitting all top trees ...")
        self._build_top_trees(generator)
        self._timers[self.TKEY_TOP_TREE].stop()
        
        self._timers[self.TKEY_DISTR_PATTS].start()
        self._logger.debug("(III) Distributing all patterns to leaves ...")
        self._distribute_all_patterns(generator)
        self._timers[self.TKEY_DISTR_PATTS].stop()
        
        self._timers[self.TKEY_BOTTOM_TREES].start()
        self._logger.debug("(IV) Fitting bottom trees ...")
        self._build_bottom_trees(generator)
        self._timers[self.TKEY_BOTTOM_TREES].stop()
        
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
            return predict_array(X, self.n_estimators, self.n_estimators_bottom, self._numpy_dtype_float, self.odir, self.store, self.wrapped_instance, self.n_jobs)
        else:
            return self._predict_generator(generator)    
    
    def get_training_times(self):

        times = {}
        
        times['total'] = self._timers[self.TKEY_ALL_FIT].get_elapsed_time()
        times['retrieve'] = self._timers[self.TKEY_TOP_SUBSETS].get_elapsed_time()
        times['top'] = self._timers[self.TKEY_TOP_TREE].get_elapsed_time()
        times['distribute'] = self._timers[self.TKEY_DISTR_PATTS].get_elapsed_time()
        times['bottom'] = self._timers[self.TKEY_BOTTOM_TREES].get_elapsed_time()
        
        return times
    
    def cleanup(self):
        
        try:
            shutil.rmtree(self.odir, ignore_errors=True)
        except:
            pass         
         
    def _retrieve_top_subsets(self, generator):
        """
        """
        
        self._logger.debug("Retrieving random subsets for all estimators...")
        seed_top = self._randomgen.randint(0, self.MAX_RAND_INT)
        subsets = generator.get_multiple_random_subsets(self.n_estimators, self.n_top, seed=seed_top)
        
        self._logger.debug("Storing subsets for all estimators ...")        
        for b in xrange(self.n_estimators):
            cname = os.path.join(self.odir, str(b), "topsub.h5")
            Xtop, ytop = subsets[b]
            ytop = ytop.reshape((len(ytop), 1))
            self.store.create_dataset(cname, "X", Xtop)
            self.store.create_dataset(cname, "y", ytop)
            
    def _build_top_trees(self, generator):
        """ Build top trees for small, random subsets 
        of the data (in-place)
        """
        
        for b in xrange(self.n_estimators):

            self._logger.debug("Fitting top tree for estimator %i ..." % b)
            toptree = self._fit_top_tree(generator, estimator_id=b)
            self._logger.debug("Saving top tree for estimator %i ..." % b)
            self.store.save(os.path.join(self.odir, str(int(b)), "toptree.tree"), toptree)

    def _distribute_all_patterns(self, generator):       
        """ Distribute points for all top trees. For each leaf of
        each top tree, an associated leaf subset is generated
        (in-place, store output)
        """
        
        generator.reset()
        
        toptrees_node_stats = {}
        for b in xrange(self.n_estimators):
            toptrees_node_stats[b] = {}
        
        while True:
            
            # process all chunks only no data are left
            
            try:
                self._logger.debug("Loading chunk of data from store (%i elements) ..." % generator.chunksize)
                X_chunk, y_chunk = self._get_next_chunk(generator)
                labels = sorted(set(list(y_chunk)))
                self._logger.info("Labels=%s" % str(labels))
            except:
                break
            
            if len(y_chunk) > 0:        
                
                params_parallel = [] 
                for b in xrange(self.n_estimators):
                    
                    self._logger.debug("\tDistributing patterns for estimator %i ..." % b)
                    odir = os.path.join(self.odir, str(b))
                    toptree = self.store.load(os.path.join(odir, "toptree.tree"), Wood)
                    args = [X_chunk, y_chunk, toptree, odir, self.store, None, None] #, toptrees_node_stats[b], self._logger
                    params_parallel.append(args)

                # FIXME: This does not work with memory stores!
                if type(self.store) == DiskStore: 
                    perform_task_in_parallel(distribute_all_patterns, params_parallel, n_jobs=self.n_jobs, backend="multiprocessing")
                else:
                    for param in params_parallel:
                        distribute_all_patterns(param)
        
        if "top" in self.plot_intermediate.keys():
            for b in xrange(self.n_estimators):
                self._logger.debug("Plotting top tree for estimator %i ..." % b)
                odir = os.path.join(self.odir, str(b))
                toptree = self.store.load(os.path.join(odir, "toptree.tree"), Wood)
                toptree.draw_tree(0, fname=os.path.join(odir, "toptree.pdf"), node_stats=toptrees_node_stats[b], **self.plot_intermediate['top'])
    
    def _build_bottom_trees(self, generator):
        """ Build all bottom trees
        """
        
        for b in xrange(self.n_estimators):
            
            odir = os.path.join(self.odir, str(int(b)))
        
            # (3) sanity check: some of the leaves might to be to big, e.g.,
            # in case a pure leaf gets filled and also contains very few
            # patterns from other classes (in this case, one still needs to 
            # build a tree but with a potentially quite large number of patterns
            #toptree = Wood.load(os.path.join(odir, "toptree.tree"))
            #self._timers[self.TKEY_SANITY_CHECK_LEAVES].start()
            #self._handle_heavy_leaves(toptree, odir, generator, b)
            #self._timers[self.TKEY_SANITY_CHECK_LEAVES].stop()
                                
            # (4) build bottom trees for each subset
            
            seed = self._randomgen.randint(0, self.MAX_RAND_INT)
            self._logger.info("Fitting bottom trees for estimator %i using seed %i ..." % (b, seed))
            
            params = self.get_params()
            args = [odir, params, self.store, self._logger, seed, self.n_jobs]
            fit_all_attached_bottom_trees(*args)
      
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
            # the new process generates copies of all parameters, 
            # we need to reassign allpreds for this reason.            
            args = [X_chunk, self.n_estimators, self.n_estimators_bottom, self._numpy_dtype_float, self.odir, self.store, self.wrapped_instance, self.n_jobs]
            #preds_chunk = start_via_single_process(predict_array, args, {})
            preds_chunk = predict_array(*args)
            
            # FIXME: This might eventually also lead to memory problems ... . Append to store as well?
            preds = numpy.concatenate([preds, preds_chunk], axis=0)
                
        return preds
    
    def _log_fitting_stats(self):
        
        total = self._timers[self.TKEY_ALL_FIT].get_elapsed_time()
        percent_retrieve = 100 * (self._timers[self.TKEY_TOP_SUBSETS].get_elapsed_time()) / total
        percent_top = 100 * (self._timers[self.TKEY_TOP_TREE].get_elapsed_time()) / total
        percent_distribute = 100 * (self._timers[self.TKEY_DISTR_PATTS].get_elapsed_time()) / total
        percent_bottom = 100 * (self._timers[self.TKEY_BOTTOM_TREES].get_elapsed_time()) / total
        
        self._logger.debug("------------------------------------------------------------------------------")
        self._logger.debug("Fitting Statistics")
        self._logger.debug("------------------------------------------------------------------------------")
        self._logger.debug("(I)\tRetrieving subsets: \t\t\t%.3f (s)\t[%2.2f %%]" % (self._timers[self.TKEY_TOP_SUBSETS].get_elapsed_time(), percent_retrieve))
        self._logger.debug("(II)\tTop tree constructions: \t\t%.3f (s)\t[%2.2f %%]" % (self._timers[self.TKEY_TOP_TREE].get_elapsed_time(), percent_top))
        self._logger.debug("(III)\tDistributing to top tree leaves: \t%.3f (s)\t[%2.2f %%]" % (self._timers[self.TKEY_DISTR_PATTS].get_elapsed_time(), percent_distribute))
        self._logger.debug("(IV)\tBottom trees constructions: \t\t%.3f (s)\t[%2.2f %%]" % (self._timers[self.TKEY_BOTTOM_TREES].get_elapsed_time(), percent_bottom))
        self._logger.debug("\t\t\t\t\t\t\t%.3f (s)\t[%2.2f %%]" % (self._timers[self.TKEY_ALL_FIT].get_elapsed_time(), 100))
        self._logger.debug("------------------------------------------------------------------------------")
        
    def _fit_top_tree(self, generator, estimator_id=0):
        """
        """

        Xshape, _ = generator.get_shapes()
        n_all = Xshape[0]     
        
        self._logger.debug("Loading random subset ...")
        cname = os.path.join(self.odir, str(estimator_id), "topsub.h5")
        Xtop = self.store.get_dataset(cname, "X")
        ytop = self.store.get_dataset(cname, "y")
        ytop = ytop.reshape(ytop.shape[0])
        
        # We want to have, in each leaf, at most n_patterns_leaf points. The problem
        # is that the top tree has to be constructed such that the number of all
        # distributed points fits the requirement above. Hence, we have to estimate
        # this number based on the points given for the top tree.
        ratio = float(self.n_top) / float(n_all)
        min_samples_split = max(1, int(self.n_patterns_leaf * ratio))
        
        # fit top tree
        self._logger.debug("Instantiating top tree ...")
        seed = self._randomgen.randint(0, self.MAX_RAND_INT)
        toptree = instantiate_single_tree_instance(self.wrapped_instance, 
                                                         seed, 
                                                         min_samples_split=min_samples_split,
                                                         top_tree_max_depth=self.top_tree_max_depth, 
                                                         balanced_top_tree=self.balanced_top_tree,
                                                         top_tree_lambda=self.top_tree_lambda,
                                                         top_tree_type=self.top_tree_type,
                                                         top_tree_leaf_stopping_mode=self.top_tree_leaf_stopping_mode,
                                                         typ="top")
    
        self._logger.debug("Fitting top tree ...")
        toptree.fit(Xtop, ytop)
        
        return toptree            
    
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
    
#     def _reduce_heavy_leaf(self, fname_store, leaf_id, toptree):
#         
#         self._logger.warning("Fitting tree for big leaf with leaf_id %i ..." % leaf_id)
#         seed = self._randomgen.randint(0, self.MAX_RAND_INT)
#         
#         numpy.random.seed(seed)
#         
#         with h5py.File(fname_store, 'r') as store_local:
#             
#             # get subset of data                
#             dset = store_local.get(leaf_id)    
#             Xsub, ysub, leaf_pure = get_XY_subsets_from_store(dset, self.heavy_leaf_domsize)
#             
#             #train subtree on subset of data
#             subtree = instantiate_single_tree_instance(self.wrapped_instance, 
#                                                              seed, 
#                                                              min_samples_split=2,
#                                                              top_tree_max_depth=3, 
#                                                              balanced_top_tree=self.balanced_top_tree,
#                                                              typ="top")
#             subtree.fit(Xsub, ysub)
# 
#             # attach subtree to toptree                        
#             toptree.attach_subtree(0, leaf_id, subtree, 0)
#             
#         return toptree
#         
#     def _chunk_too_large(self, n, d):
#         
#         # assuming double precision here
#         return n * d * 8 > self.chunk_max_megabytes * 1000000
#         
#     def _handle_heavy_leaves(self, toptree, odir, generator, estimator_id):
#         """
#         """
#         
#         fname_store = os.path.join(odir, "data.h5")        
#         with h5py.File(fname_store, 'r') as store:
#             unique_leaf_ids = store.keys()
#         
#         heavy_leaves_detected = False
#         
#         for i in xrange(len(unique_leaf_ids)):
#                     
#             leaf_id = unique_leaf_ids[i]
#     
#             with h5py.File(fname_store, 'r') as store_local:
#                 
#                 dset = store_local.get(leaf_id)
#                 self._logger.debug("Leaf with leaf_id %s has size:\t%i" % (str(leaf_id), dset.shape[0]))
#                 if self._chunk_too_large(dset.shape[0], dset.shape[1]):
#                     self._logger.debug("Reducing heavy leaf with leaf_id %s ..." % str(leaf_id))
#                     toptree = self._reduce_heavy_leaf(fname_store, leaf_id, toptree)
#                     heavy_leaves_detected = True
#                     
#         # redistribute points to subsets if needed
#         if heavy_leaves_detected == True:
#             self._logger.debug("Redistributing patterns for estimator %i (due to heavy leaves) ..." % estimator_id)
#             toptree_node_stats = distribute_all_patterns(generator, toptree, odir)
#             if "top_adapted" in self.plot_intermediate.keys():
#                 self._logger.debug("Replotting top tree for estimator %i ..." % estimator_id)
#                 toptree.draw_tree(0, fname=os.path.join(odir, "toptree_adapted.pdf"), node_stats=toptree_node_stats, **self.plot_intermediate['top'])
#     
#             self._logger.debug("Saving modified top tree for estimator %i ..." % estimator_id)
#             toptree.save(os.path.join(odir, "toptree.tree"))