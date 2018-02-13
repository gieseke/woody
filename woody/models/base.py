#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

from .util import init_logger

class NoLogger():
    
    def __init__(self):
        pass
    
    def info(self, msg):
        pass
    
    def debug(self, msg):
        pass
    
class BaseEstimator(object):
    
    def __init__(self,
                 verbose=0,
                 logging_name="BaseEstimator",
                 logging_file=None,
                 seed=0,
                 ):
        
        self.verbose = verbose
        self.logging_name = logging_name
        self.seed = seed
        
    def fit(self, logging_file="estimator.log"):
        
        # instantiate logger
        if self.verbose > 0:
            self._logger = init_logger(fname=logging_file,
                                       log_name=self.logging_name,
                                       log_level="DEBUG")
        else:
            self._logger = NoLogger()
           
    def get_params(self):
        """ Returns the models's parameters
        """
        
        return {"verbose": self.verbose,
                "logging_name" : self.logging_name,
                "seed": self.seed,
                }
        
    def set_params(self, **parameters):
        """ Sets local parameters (does not need
        to be overwritten).
        """
        
        for parameter, value in parameters.items():
            self.setattr(parameter, value)               
        

