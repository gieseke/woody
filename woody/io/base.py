#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

from .csv import CSVReader
from .h5 import H5Reader
from .h5pd import H5PandasReader
from .mem import MemoryReader

class DataGenerator(object):
    """
    """
            
    def __init__(self,
                 fname=None,
                 data=None, 
                 patterns=True,
                 target=True,
                 chunksize=32000,                
                 n_lines_max=None, 
                 target_column=None,
                 patterns_columns=None,
                 seed=0,
                 parsing_args={}
                 ):
        
        if data is not None:
            
            self._reader = MemoryReader(data=data,
                                     patterns=patterns,
                                     target=patterns,
                                     chunksize=1000000,
                                     seed=seed)
            
        elif fname.endswith(".csv") or fname.endswith(".csv.gz"):
            
            self._reader = CSVReader(fname, 
                                     patterns=patterns,
                                     target=target,
                                     chunksize=chunksize,
                                     target_column=target_column,
                                     patterns_columns=patterns_columns,
                                     seed=seed,
                                     parsing_args=parsing_args,
                                     )
            
        elif fname.endswith(".h5pd"):
            
            self._reader = H5PandasReader(fname, 
                                          patterns=patterns,
                                          target=target,
                                          chunksize=chunksize,
                                          n_lines_max=n_lines_max,                                    
                                          seed=seed,
                                          )                                                    
        elif fname.endswith(".h5"):
                
            self._reader = H5Reader(fname, 
                                    patterns=patterns,
                                    target=target,
                                    chunksize=chunksize,                                    
                                    seed=seed,
                                    )
                                                
        else:
            raise Exception("Unknown file extension for file %s. Cannot instantiate reader!" % fname)
        
        self.reset()
        self.chunksize = chunksize
            
    def __del__(self):
        
        self.close()

    def close(self):
        """
        """
                
        self._reader.close()
        
    def set_seed(self, s):
        
        self._reader.set_seed(s)
        
    def set_mode(self, patterns=True, target=True):
        """
        """
        
        self._reader.set_mode(patterns=patterns, target=target)       
                                
    def reset(self):
        """
        """
        
        self._reader.reset()
        
    def get_random_subset(self, size, chunk_percent=0.5, shuffle=True):
        """
        """
        
        return self._reader.get_random_subset(size, 
                                              chunk_percent=chunk_percent, 
                                              shuffle=shuffle,
                                              )
        
    def get_multiple_random_subsets(self, nsubs, size, chunk_percent=None, shuffle=True, seed=None):

        return self._reader.get_multiple_random_subsets(nsubs, 
                                                        size, 
                                                        chunk_percent=chunk_percent, 
                                                        shuffle=shuffle,
                                                        seed=seed,
                                                        )                
        
    def get_chunk(self, extract=True):
        """
        """
        
        return self._reader.get_chunk(extract=extract)
        
    def get_all(self):
        """
        """
        
        return self._reader.get_all()
    
    def to_csv(self, fname, cache=False, remove=False):
        
        self._reader.to_csv(fname, cache=cache, remove=remove)
        
    def get_all_patterns(self):
        
        self._reader.set_mode(patterns=True, target=False)
        patterns = self.get_all()
        
        return patterns
            
    def get_all_target(self):
        
        self._reader.set_mode(patterns=False, target=True)
        target = self.get_all()
        
        return target
    
    def get_shapes(self):
        
        return self._reader.get_shapes()
    
    