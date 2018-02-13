#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import numpy
import pandas

from .reader import Reader

class H5PandasReader(Reader):
    """
    """
        
    def __init__(self,
                 fname, 
                 patterns=True,
                 target=True,
                 chunksize=1000000,
                 n_lines_max=None,
                 seed=0,
                 ):

        super(H5PandasReader, self).__init__(fname=fname, 
                                       patterns=patterns,
                                       target=target,
                                       chunksize=chunksize,
                                       n_lines_max=n_lines_max,
                                       seed=seed)
    
                    
        if self.n_lines_max is not None:
            print("n_lines_max is set to %i " % (self.n_lines_max))
        
    def __del__(self):
        
        super(H5PandasReader, self).__del__()
        
        try:
            self._store.close()
        except:
            pass        
                                        
    def reset(self):
        
        try:
            self._store.close()
        except:
            pass
        
        self._store = pandas.HDFStore(self.fname, mode='r')
        self._chunk_counter = 0
        
    def get_chunk(self, extract=True):
        
        n_lines = self.get_n_lines_max()

        if self._chunk_counter > n_lines:
            raise Exception("Reached end of data!")

        start = self._chunk_counter
        end = min(n_lines, self._chunk_counter + self.chunksize)
        self._chunk_counter += self.chunksize
                
        if self.patterns == True and self.target == True:
            data_chunk_X = self._store.select('X', start=start, stop=end)
            data_chunk_y = self._store.select('y', start=start, stop=end)
            data_chunk = numpy.concatenate((data_chunk_X, data_chunk_y), axis=1)
        elif self.patterns == True:            
            data_chunk = self._store.select('X', start=start, stop=end)

        elif self.target == True:
            data_chunk = self._store.select('y', start=start, stop=end)

        if extract == True:
            data_chunk = self._get_patterns_labels(data_chunk)
            
        return data_chunk
    
    def get_random_subset(self, size, chunk_percent=None, shuffle=True):
        
        nrows = self.get_n_lines_max()
        if size < nrows:
            # retrieving the (random) items in sorted order is faster
            choice = numpy.sort(numpy.array(self._randomgen.sample(xrange(0, nrows), size)))
        else:
            choice = numpy.array(range(0, nrows))
        
        if self.patterns == True and self.target == True:
            data_chunk_X = self._store.select('X', where=pandas.Index(choice))
            data_chunk_Y = self._store.select('y', where=pandas.Index(choice))
            data_chunk = numpy.concatenate((data_chunk_X, data_chunk_Y), axis=1)
        elif self.patterns == True:
            data_chunk = self._store.select('X', where=pandas.Index(choice))
        elif self.target == True:
            data_chunk = self._store.select('y', where=pandas.Index(choice))
        
        return self._get_patterns_labels(data_chunk)
    
    def get_multiple_random_subsets(self, nsubs, size, chunk_percent=None, shuffle=True, seed=None):
        
        if seed is not None:
            self._randomgen.seed(seed)
            
        nrows = self.get_n_lines_max()
        
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
            data_chunk_X = self._store.select('X', where=pandas.Index(union_choices))
            data_chunk_Y = self._store.select('y', where=pandas.Index(union_choices))
            data_chunk = numpy.concatenate((data_chunk_X, data_chunk_Y), axis=1)
        elif self.patterns == True:
            data_chunk = self._store.select('X', where=pandas.Index(union_choices))
        elif self.target == True:
            data_chunk = self._store.select('y', where=pandas.Index(union_choices))
            
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
        
        n_rows = self.get_n_lines_max()
                
        if self.patterns == True and self.target == True:
            X = self._store.select('X', stop=n_rows)
            y = self._store.select('y', stop=n_rows)
            data = numpy.concatenate((X, y), axis=1)
        elif self.patterns == True:            
            data = self._store.select('X', stop=n_rows)
        elif self.target == True:
            data = self._store.select('y', stop=n_rows)
        
        data = data[:n_rows]
        return self._get_patterns_labels(data)
    
    def get_shapes(self):
        
        n_rows = self.get_n_lines_max()
        
        if self.patterns == True and self.target == True:
            
            ncols_X = self._store.get_storer('X').ncols
            ncols_y = self._store.get_storer('y').ncols
            
            return (n_rows, ncols_X), (n_rows, ncols_y)
        
        elif self.patterns == True:
            
            ncols_X = self._store.get_storer('X').ncols
            
            return (n_rows, ncols_X)
        
        elif self.target == True:
            
            ncols_y = self._store.get_storer('y').ncols
            
            return (n_rows, ncols_y)     
        
        raise Exception("Either patterns or target must be True!")
    
    def get_n_lines_max(self):

        if self.patterns == True:
            n_lines = self._store.get_storer('X').nrows
        elif self.target == True: 
            n_lines = self._store.get_storer('y').nrows
                    
        if self.n_lines_max is not None:
            n_lines = min(n_lines, self.n_lines_max)

        return n_lines
    
    def to_csv(self, fname, cache=False, remove=True):
        
        self.reset()
                
        if cache == True and os.path.isfile(fname):
            return

        if remove == True:
            try:
                os.remove(fname)
            except:
                pass        
            
        while True:
            try:
                chunk = self.get_chunk(extract=False)
                df = pandas.DataFrame(chunk, index=range(0, chunk.shape[0]))
                print df.head(1).to_string()                
                df.to_csv(fname, mode="a", sep=",", header=False, index=False)
            except:
                break
        
            
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
    