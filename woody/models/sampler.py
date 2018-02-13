#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import copy
import numpy

class Sampler(object):
    """
    """

    def __init__(self, model, seed=0, n_estimators=10, percentage=0.5):
        
        self.model = model
        self.seed = seed
        self.n_estimators = n_estimators
        self.percentage = percentage
        
        self.models = []
        for i in xrange(self.n_estimators):
            self.models.append(copy.deepcopy(self.model))
             
    def fit(self, X, y):
        
        for i in xrange(self.n_estimators):
            print("Fitting model %i ..." % i) 
            partition = numpy.random.permutation(X.shape[0])
            partition = partition[:int(self.percentage * len(partition))]
            Xsub = X[partition]
            ysub = y[partition]
            self.models[i].fit(Xsub, ysub)
                
    def predict(self, X, operator="max"):
        
        all_predictions = self._predict_all(X)
        
        preds = []
        for j in xrange(all_predictions.shape[0]):
            p = all_predictions[j,:]
            values, counts = numpy.unique(p,return_counts=True)
            ind = numpy.argmax(counts)
            preds.append(values[ind])
        preds = numpy.array(preds)
        
        return preds
    
    def _predict_all(self, X):
        
        predictions = []
        for i in xrange(self.n_estimators):
            print("Computing predictions for model %i ..." % i)
            preds = self.models[i].predict(X)
            predictions.append(preds)
        predictions = numpy.array(predictions).T
        
        return predictions    
        