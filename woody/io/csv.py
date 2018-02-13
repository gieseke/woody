#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import numpy
import pandas

from .reader import Reader

class CSVReader(Reader):
    """
    """
        
    def __init__(self,
                 fname, 
                 patterns=True,
                 target=True,
                 chunksize=32000,                 
                 target_column=None,
                 patterns_columns=None,
                 seed=0,
                 parsing_args={}
                 ):
        
        super(CSVReader, self).__init__(fname=fname, 
                                       patterns=patterns,
                                       target=target,
                                       chunksize=chunksize,                                       
                                       seed=seed)

        self.target_column = target_column
        self.patterns_columns = patterns_columns
        self.parsing_args = parsing_args
                        
    def reset(self):
        
        self.close()
                
        self._reader = pandas.read_csv(self.fname, iterator=True, chunksize=self.chunksize, **self.parsing_args)
            
    def get_random_subset(self, size, chunk_percent=0.5, shuffle=True):
        """
        NOTE: Seems to interfer with yep (multiprocessing, deadlock?)
        
        """

        data = None
        
        rand_per_chunk = int(self.chunksize * chunk_percent)
    
        while data is None or len(data) < size:

            self.reset(self.chunksize)
            
            for chunk in self._reader:
                
                data_chunk = self._transform_csv(chunk)
                choice = sorted(self._randomgen.sample(xrange(len(data_chunk)), rand_per_chunk))
                data_chunk = data_chunk[choice]
                
                if data is None:
                    data = data_chunk  
                else:
                    data = numpy.concatenate((data, data_chunk), axis=0)
                
                if len(data) >= size:
                    break                
                
            self.close()
        
        if shuffle == True:
            partition = range(len(data))
            self._randomgen.shuffle(partition)
            data = data[partition]
        data = data[:size]

        return self._get_patterns_labels(data)
    
    def get_chunk(self, extract=True):
        
        chunk = self._reader.get_chunk()
        data_chunk = self._transform_csv(chunk)

        if extract == True:
            data_chunk = self._get_patterns_labels(data_chunk)
                    
        return data_chunk
    
    def transform(self, chunk):
        
        return chunk.ix[:,:].values
            
    def _get_patterns_labels(self, data):
        
        if self.patterns == True and self.target == True:
                                                                              
            X = numpy.ascontiguousarray(data[:, self.patterns_columns])
            y = numpy.ascontiguousarray(data[:, self.target_column])
            return X, y
        
        elif self.patterns == True:
            
            X = numpy.ascontiguousarray(data[:, self.patterns_columns])
            return X
        
        elif self.target == True:
            
            y = numpy.ascontiguousarray(data[:, self.target_column])
            return y   
        
        raise Exception("Both patterns and target is set to False!")
    
    
    def get_all(self):
        
        self.reset()
          
        data = None
        
        while True:
            
            try:
                
                data_chunk = self.get_chunk(extract=False)
                
            except Exception as e:
                break

            if data is None:
                data = data_chunk  
            else:
                data = numpy.concatenate((data, data_chunk), axis=0)               
        
        return self._get_patterns_labels(data)  
            