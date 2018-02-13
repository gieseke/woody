#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import numpy

from .reader import Reader

class MemoryReader(Reader):
    """
    """
        
    def __init__(self,
                 data, 
                 patterns=True,
                 target=True,
                 chunksize=1000000,
                 seed=0,
                 ):

        super(MemoryReader, self).__init__(data=data, 
                                       patterns=patterns,
                                       target=target,
                                       chunksize=chunksize,
                                       seed=seed)
        
        if self.data['y'].ndim == 1:
            self.data['y'] = self.data['y'].reshape((len(self.data['y']), 1))
        
    def __del__(self):
        
        super(MemoryReader, self).__del__()
                                                
    def reset(self):
        
        self._store = {}
        self._chunk_counter = 0
        
    def get_chunk(self, extract=True):
        
        if self.patterns == True:
            n_lines = len(self.data['X'])
        elif self.target == True: 
            n_lines = len(self.data['y'])

        if self._chunk_counter > n_lines:
            raise Exception("Reached end of data!")

        start = self._chunk_counter
        end = min(n_lines, self._chunk_counter + self.chunksize)
        self._chunk_counter += self.chunksize
                
        if self.patterns == True and self.target == True:
            data_chunk_X = self.data['X'][start:end,:]
            data_chunk_y = self.data['y'][start:end, :]
            data_chunk = numpy.concatenate((data_chunk_X, data_chunk_y), axis=1)
        elif self.patterns == True:            
            data_chunk = self.data['X'][start:end,:]

        elif self.target == True:
            data_chunk = self.data['y'][start:end,:]

        if extract == True:
            data_chunk = self._get_patterns_labels(data_chunk)
            
        return data_chunk
    
    def get_random_subset(self, size, chunk_percent=None, shuffle=True):
        
        nrows = len(self.data['X'])
        if size < nrows:
            # retrieving the (random) items in sorted order is faster
            choice = numpy.sort(numpy.array(self._randomgen.sample(xrange(0, nrows), size)))
        else:
            choice = numpy.array(range(0, nrows))
        
        if self.patterns == True and self.target == True:
            data_chunk_X = self.data['X'][choice,:]
            data_chunk_Y = self.data['y'][choice,:]
            data_chunk = numpy.concatenate((data_chunk_X, data_chunk_Y), axis=1)
        elif self.patterns == True:
            data_chunk = self.data['X'][choice,:]
        elif self.target == True:
            data_chunk = self.data['y'][choice,:]
        
        return self._get_patterns_labels(data_chunk)
    
    def get_multiple_random_subsets(self, nsubs, size, chunk_percent=None, shuffle=True, seed=None):
        
        if seed is not None:
            self._randomgen.seed(seed)
                    
        nrows = len(self.data['X'])
        if size < nrows:
            mult_choices = [numpy.sort(numpy.array(self._randomgen.sample(xrange(0, nrows), size))) for _ in xrange(nsubs)]
        else:
            mult_choices = [numpy.array(range(0, nrows)) for i in range(nsubs)]
        
        union_choices = numpy.zeros(0, dtype=numpy.int64)
        for i in xrange(nsubs):
            union_choices = numpy.union1d(union_choices, mult_choices[i])
        union_choices = numpy.sort(union_choices)
        
        # could cover a very broad range of numbers, hence
        # use a dictionary to store inverse mappings
        union_choices_mappings = {}
        for j in xrange(len(union_choices)):
            union_choices_mappings[union_choices[j]] = j
        
        if self.patterns == True and self.target == True:
            data_chunk_X = self.data['X'][union_choices,:]
            data_chunk_Y = self.data['y'][union_choices,:]
            data_chunk = numpy.concatenate((data_chunk_X, data_chunk_Y), axis=1)
        elif self.patterns == True:
            data_chunk = self.data['X'][union_choices,:]
        elif self.target == True:
            data_chunk = self.data['y'][union_choices,:]
            
        # retrieve individual random subsets
        data_patterns_labels = []
        for i in xrange(nsubs):
            choice = mult_choices[i]
            mapped_indices = []
            for j in xrange(size):
                mapped_indices.append(union_choices_mappings[choice[j]])
            mapped_indices = numpy.array(mapped_indices)
            data_patterns_labels.append(self._get_patterns_labels(data_chunk[mapped_indices, :])) 
        
        return data_patterns_labels
        
    def get_all(self):
        
        if self.patterns == True and self.target == True:
            X = self.data['X']
            y = self.data['y']
            data = numpy.concatenate((X, y), axis=1)
        elif self.patterns == True:            
            data = self.data['X']
        elif self.target == True:
            data = self.data['y']
            
        return self._get_patterns_labels(data)
    
    def get_shapes(self):
        
        if self.patterns == True and self.target == True:
            return self.data['X'].shape, self.data['y'].shape
        
        elif self.patterns == True:
            return self.data['X'].shape
        
        elif self.target == True:
            return self.data['y'].shape 
        
        raise Exception("Either patterns or target must be True!")
    
    def _get_patterns_labels(self, data):
        
        if self.patterns == True and self.target == True:
            
            X = numpy.ascontiguousarray(data[:,:-1])
            y = numpy.ascontiguousarray(data[:, -1])
            return X, y
        
        elif self.patterns == True:
            
            X = numpy.ascontiguousarray(data)
            return X
        
        elif self.target == True:
            
            y = numpy.ascontiguousarray(data)
            y = y.reshape(len(y))
            return y
        
        raise Exception("Both patterns and target is set to False!")    
    