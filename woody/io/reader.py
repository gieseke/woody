#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import random

class Reader(object):
    """
    """
        
    def __init__(self,
                 fname=None,
                 data=None, 
                 patterns=True,
                 target=True,
                 chunksize=32000,        
                 n_lines_max=None,         
                 seed=0,
                 ):
        
        self.fname = fname
        self.data = data
        self.patterns = patterns
        self.target = target
        self.chunksize = chunksize
        self.n_lines_max = n_lines_max
        self.seed = seed    
        
        self._randomgen = random.Random(self.seed)
        self._reader = None
    
    def __del__(self):
        
        self.close()
            
    def close(self):
        
        try:
            self._reader.close()
        except:
            pass
                
    def set_seed(self, s):
        
        self._randomgen.seed(s)
        
    def set_mode(self, patterns=True, target=True):
        
        self.patterns = patterns
        self.target = target
        