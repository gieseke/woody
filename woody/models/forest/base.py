#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import abc
import math
import numpy as np
import cPickle as pickle

from .util import PickableWoodyRFWrapper, ensure_data_types
from woody.util.array import transpose_array
from woody.util import draw_single_tree
            
class Wood(object):
    """
    Random forest implementation.
    """
    __metaclass__ = abc.ABCMeta

    ALLOWED_FLOAT_TYPES = ['float',
                           'double',
                           ]
    TREE_TRAVERSAL_MODE_MAP = {"dfs": 0,
                               "node_size": 1,
                               "prob": 2,
                               }
    CRITERION_MAP = {"mse": 0,
                     "gini": 1,
                     "entropy": 2,
                     "even_mse": 3,
                     "even_gini": 3,
                     "even_entropy": 3,
                     }
    LEARNING_TYPE_MAP = {"regression": 0,
                         "classification": 1,
                         }
    TREE_TYPE_MAP = {"standard":0,
                     "randomized":1,
                     }
    LEAF_STOP_MODE_MAP = {"all":0,
                          "ignore_impurity":1,
                          }
    TRANSPOSED_MAP = {False: 0,
                      True: 1,
                      }
    
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
                 lam_criterion = 0.0, 
                 n_jobs=1,                                  
                 verbose=1,
                 ):

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
        self.patts_trans = self.TRANSPOSED_MAP[patts_trans]
        self.do_patts_trans = do_patts_trans
        self.lam_criterion = lam_criterion
        self.n_jobs = n_jobs
        self.verbose = verbose
                
        # set numpy float and int dtypes
        if self.float_type == "float":
            self.numpy_dtype_float = np.float32
        else:
            self.numpy_dtype_float = np.float64
        self.numpy_dtype_int = np.int32
                        
        assert self.float_type in self.ALLOWED_FLOAT_TYPES
        
    def __del__(self):
        """ Destructor taking care of freeing
        internal and external (Swig) resources.
        """

        if hasattr(self, 'wrapper_params'):
            
            self.wrapper.module.free_resources_extern(self.wrapper.params,
                                                      self.wrapper.forest)

    def get_params(self, deep=True):
        
        return {"seed": self.seed, 
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
                }

    def set_params(self, **parameters):
        """ Updates local parameters
        """
        
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        
    def fit(self, X, y, indices=None):
        """ If indices is not None, then 
        consider X[indices] instead of X 
        (in-place).
        """

        if X.shape[0] != y.shape[0]:
            raise ValueError("Dimensions not equal:" 
                             "X.shape[0]=%i != y.shape[0]=%i" %
                             (X.shape[0], y.shape[0]))
            
        #if self.tree_type == "standard" and self.bootstrap == False:
        #    raise Exception("No randomness given: bootstrap=%s and tree_type=%s" % (str(self.bootstrap), str(self.tree_type)))

        # convert input data to correct types and generate local
        # copies to prevent destruction of objects
        X, y = ensure_data_types(X, y, self.numpy_dtype_float)
        
        # transform some parameters
        if self.max_features == None:
            max_features = X.shape[1]
        elif isinstance(self.max_features, int):
            if self.max_features < 1 or self.max_features > X.shape[1]:
                raise Exception("max.features=%i must "
                                "be >= 1 and <= X.shape[1]=%i" % 
                                (self.max_features, X.shape[1]))
            max_features = self.max_features
        elif self.max_features == "sqrt":
            max_features = int(math.sqrt(X.shape[1]))
        elif self.max_features == "log2":
            max_features = int(math.log(X.shape[1]), 2)
        else:
            max_features = 1

        # set max_depth
        max_depth = ((2 ** 31) - 1 if self.max_depth is None else self.max_depth)

        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than zero!")
    
        self.wrapper = PickableWoodyRFWrapper(self.float_type)
        
        if self.do_patts_trans == True:
            XT = np.empty(X.shape, dtype=X.dtype)
            transpose_array(X, XT)
            X = XT

        self.wrapper.module.init_extern(self.seed, 
                                        self.n_estimators, 
                                        self.min_samples_split, 
                                        max_features, 
                                        self.bootstrap, 
                                        max_depth, 
                                        self.min_samples_leaf, 
                                        self.LEARNING_TYPE_MAP[self.learning_type], 
                                        self.CRITERION_MAP[self.criterion], 
                                        self.TREE_TRAVERSAL_MODE_MAP[self.tree_traversal_mode], 
                                        self.LEAF_STOP_MODE_MAP[self.leaf_stopping_mode],
                                        self.TREE_TYPE_MAP[self.tree_type], 
                                        self.n_jobs, 
                                        self.verbose, 
                                        self.patts_trans, 
                                        self.wrapper.params, 
                                        self.wrapper.forest, 
                                        )
        
        self.wrapper.params.lam_crit = self.lam_criterion
                    
        if indices is not None:
            use_indices = 1
            indices = np.array(indices).astype(dtype=np.int32)
            indices_weights = np.ones(indices.shape, dtype=np.int32)
            if indices.ndim == 1:
                indices = indices.reshape((1, len(indices)))
                indices_weights = indices_weights.reshape((1, len(indices_weights)))
                
            if (indices.shape[0] != self.n_estimators) or \
                (indices_weights.shape[0] != self.n_estimators):
                raise Exception("""
                    Both 'indices' and 'indices_weights' must be of shape 
                    (n_estimators, x), but are of shape %s and %s, respectively!
                """ % (str(indices.shape), str(indices_weights.shape)))
                                
        else:
            # dummy parameters
            use_indices = 0
            indices = np.empty((0, 0), dtype=np.int32)
            indices_weights = np.empty((0, 0), dtype=np.int32)
        
        self.wrapper.module.fit_extern(X, y, indices, indices_weights, use_indices, self.wrapper.params, self.wrapper.forest)

        return self

    def predict(self, X, indices=None):
        """
        """
        
        if X.dtype != self.numpy_dtype_float:
            X = X.astype(self.numpy_dtype_float)      
        
        if indices is None: 
            indices = np.empty((0, 0), dtype=np.int32)
        else:
            indices = np.array(indices).astype(dtype=np.int32)
            if indices.ndim == 1:
                indices = indices.reshape((1, len(indices)))   
                
        preds = np.ones(X.shape[0], dtype=self.numpy_dtype_float)
        
        self.wrapper.module.predict_extern(X, preds, indices, self.wrapper.params, self.wrapper.forest)

        return preds

    def predict_all(self, X, indices=None):
        """
        """
        
        if X.dtype != self.numpy_dtype_float:
            X = X.astype(self.numpy_dtype_float)      
        
        if indices is None: 
            indices = np.empty((0, 0), dtype=np.int32)
        else:
            indices = np.array(indices).astype(dtype=np.int32)
            if indices.ndim == 1:
                indices = indices.reshape((1, len(indices)))   
                
        preds = np.ones((X.shape[0], self.n_estimators), dtype=self.numpy_dtype_float)
        
        self.wrapper.module.predict_all_extern(X, preds, indices, self.wrapper.params, self.wrapper.forest)

        return preds

    def get_leaves_ids(self, X, n_jobs=1, indices=None, verbose=0):
        
        if X.dtype != self.numpy_dtype_float:
            X = X.astype(self.numpy_dtype_float)

        if indices is None: 
            indices = np.empty((0, 0), dtype=np.int32)
            preds = np.zeros(X.shape[0] * self.n_estimators, dtype=self.numpy_dtype_float)
            self.wrapper.params.prediction_type = 1
            self.wrapper.params.verbosity_level = verbose
            self.wrapper.module.predict_extern(X, preds, indices, self.wrapper.params, self.wrapper.forest)
        else:

            indices = np.array(indices).astype(dtype=np.int32)
            if indices.ndim == 1:
                indices = indices.reshape((1, len(indices)))            
            
            preds = np.zeros(self.n_estimators * indices.shape[1], dtype=self.numpy_dtype_float)
            self.wrapper.params.prediction_type = 1
            self.wrapper.params.verbosity_level = verbose
            
            self.wrapper.module.predict_extern(X, preds, indices, self.wrapper.params, self.wrapper.forest)            
            
        return preds
    
    def print_parameters(self):
        """
        """
        
        self.wrapper.module.print_parameters_extern(self.wrapper.params)

    def get_n_nodes(self, tree_index):
        
        tree = self.wrapper.module.TREE() 
        self.wrapper.module.get_tree_extern(tree, tree_index, self.wrapper.forest)
        n_nodes = tree.node_counter
        
        return n_nodes
    
    def get_wrapped_tree(self, index):

        tree = self.wrapper.module.TREE() 
        self.wrapper.module.get_tree_extern(tree, index, self.wrapper.forest)
        
        return tree        
        
    def get_tree(self, index):
        
        try:
            import networkx as nx
        except Exception as e:
            raise Exception("Module 'networkx' is required to export the tree structure: %s" % str(e))
        
        tree = self.get_wrapped_tree(index)
        n_nodes = tree.node_counter
        
        nodes = []
        for i in xrange(n_nodes):
            # i is also the node_id (stored consecutively)
            node = self.wrapper.module.TREE_NODE()
            self.wrapper.module.get_tree_node_extern(tree, i, node)
            nodes.append(node)
        
        G = nx.Graph()
        for i in xrange(len(nodes)):
            G.add_node(i)
            
        for i in xrange(len(nodes)):
            G.node[i]['node_id'] = i
            if nodes[i].left_id == 0 and nodes[i].right_id == 0:
                G.node[i]['is_leaf'] = True
            else:
                G.node[i]['is_leaf'] = False
            G.node[i]['leaf_criterion'] = int(nodes[i].leaf_criterion)

        for i in xrange(len(nodes)):
            if nodes[i].left_id != 0:
                G.add_edge(i, nodes[i].left_id)
            if nodes[i].right_id != 0:
                G.add_edge(i, nodes[i].right_id)

        return G

    def draw_tree(self, index, node_stats=None, ax=None, figsize=(200,20), fname="tree.pdf", with_labels=False, edge_width=1.0, edges_alpha=1.0, arrows=False, alpha=0.5, node_size=1000):
        """
        """
        
        tree = self.get_tree(index)
        draw_single_tree(tree, 
                         node_stats=node_stats,
                         ax=ax,
                         figsize=figsize,
                         fname=fname,
                         with_labels=with_labels,
                         arrows=arrows,
                         edge_width=edge_width,
                         edges_alpha=edges_alpha,
                         alpha=alpha,
                         node_size=node_size, 
                         )
                    
    def attach_subtree(self, index, leaf_id, subtree, subtree_index):
        """ Replaces the leaf with id leaf_id 
        with the subtree provided
        """
        
        wrapped_subtree = subtree.get_wrapped_tree(subtree_index)
        self.wrapper.module.attach_tree_extern(index, self.wrapper.forest, wrapped_subtree, int(leaf_id))        
    
    def save(self, fname):
        """
        Saves the model to a file.
        
        Parameters
        ----------
        fname : str
            the filename of the model
        """
        
        d = os.path.dirname(fname)
        if not os.path.exists(d):
            os.makedirs(d)            
        
        try:
            
            # protocol=0 (readable), protocol=1 (python2.3 and 
            # backward), protocol=2 (binary, new python versions
            filehandler_model = open(fname, 'wb') 
            pickle.dump(self, filehandler_model, protocol=2)
            filehandler_model.close()
            
        except Exception, e:
            
            raise Exception("Error while saving model to " + unicode(fname) + u":" + unicode(e))
                
    @staticmethod
    def load(fname):
        """
        Loads a model from disk.
          
        Parameters
        ----------
        filename_model : str
            The filename of the model
  
        Returns
        -------
        Model: the loaded model
        """
  
        try:
          
            filehandler_model = open(fname, 'rb')
            new_model = pickle.load(filehandler_model)
            filehandler_model.close()
            
            return new_model
          
        except Exception, e:
              
            raise Exception("Error while loading model from " + unicode(fname) + u":" + unicode(e))
