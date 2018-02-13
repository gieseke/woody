#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import numpy
import h5py

from .reader import Reader

class H5Reader(Reader):
    """
    """
        
    def __init__(self,
                 fname, 
                 patterns=True,
                 target=True,
                 chunksize=32000,
                 seed=0,
                 ):

        super(H5Reader, self).__init__(fname=fname, 
                                       patterns=patterns,
                                       target=target,
                                       chunksize=chunksize,
                                       seed=seed)
                                        
    def reset(self):
        
        self.close()
                
        self._reader = h5py.File(self.fname, 'r', libver='latest')
        self._reader_chunk_counter = 0
        
    def get_chunk(self, extract=True):
        
        if self.patterns == True:
            dsetX = self._reader.get("X")
            n_lines = len(dsetX)
        if self.target == True:
            dsety = self._reader.get("y")
            n_lines = len(dsety)

        if self._reader_chunk_counter > n_lines:
            raise Exception("Reached end of data!")

        start = self._reader_chunk_counter
        end = min(n_lines, self._reader_chunk_counter + self.chunksize)
        self._reader_chunk_counter += self.chunksize
        
        if self.patterns == True and self.target == True:
            data_chunk = dsetX[start:end, :]
            data_chunk = numpy.concatenate((data_chunk, dsety[start:end, :]), axis=1)
        elif self.patterns == True:            
            data_chunk = dsetX[start:end, :]

        elif self.target == True:
            data_chunk = dsety[start:end, :]

        if extract == True:
            data_chunk = self._get_patterns_labels(data_chunk)
            
        return data_chunk
    
    def get_random_subset(self, size, chunk_percent=0.5, shuffle=True):
        
        self.reset()
                        
        nrows = self._reader.get("y").shape[0]
        
        if size < nrows:
            choice = numpy.array(self._randomgen.sample(xrange(0, nrows), size))
        else:
            choice = numpy.array(range(0, nrows))
        
        data = None
        counter = 0
        
        while True:
            
            try:
                
                data_chunk = self.get_chunk(extract=False)
                choice_chunk = choice[choice < counter + len(data_chunk)] - counter
                data_choice = data_chunk[choice_chunk,:]
                
                counter += len(data_chunk)
                choice = choice[choice >= counter]
                                            
            except:
                break

            if data is None:
                data = data_choice  
            else:
                data = numpy.concatenate((data, data_choice), axis=0)
                  
                  
        return self._get_patterns_labels(data)        

        
#         data = None
#         rand_per_chunk = int(self.chunksize * chunk_percent)
#         
#         while data is None or len(data) < size:
#             
#             self.reset()
#             
#             if self.patterns == True:
#                 dsetX = self._reader.get("X")
#             if self.target == True:
#                 dsety = self._reader.get("y")
#             
#             n_lines = len(dsety)
#             chunk_counter = 0
#             
#             while chunk_counter < n_lines:
#                 
#                 start = chunk_counter
#                 end = min(n_lines, chunk_counter+self.chunksize)
#             
#                 if self.patterns == True and self.target == True:
#                     data_chunk = dsetX[start:end, :]
#                     data_chunk = numpy.concatenate((data_chunk, dsety[start:end, :]), axis=1)
#                 elif self.patterns == True:
#                     data_chunk = dsetX[start:end, :]
#                 elif self.target == True:
#                     data_chunk = dsety[start:end, :]
#                      
#                 choice = sorted(self._randomgen.sample(xrange(len(data_chunk)), rand_per_chunk))
#                 data_chunk = data_chunk[choice]
# 
#                 if data is None:
#                     data = data_chunk
#                 else:
#                     data = numpy.concatenate((data, data_chunk), axis=0)
#                 chunk_counter += self.chunksize
#                                 
#                 if len(data) >= size:
#                     break    
#                 
#             self.close() 
#                       
#         if shuffle == True:
#             partition = range(len(data))
#             self._randomgen.shuffle(partition)            
#             data = data[partition]
#         data = data[:size]
#         
#         return self._get_patterns_labels(data)
    
#     def generate_random_subsets(self, size, n_estimatos, ostore):
# 
#         self.reset()
#                         
#         nrows = self._reader.get("y").shape[0]
#         choices = []
#         for i in xrange(n_estimatos):
#             choices.append(sorted(numpy.random.randint(0, nrows, size=size)))
#           
#         data_list = [[] for _ in xrange(n_estimatos)]
#         counters = numpy.zeros(n_estimatos, dtype=numpy.int32)
#         
#         while True:
#             
#             try:
#                 
#                 data_chunk = self.get_chunk(extract=False)
#                 
#                 for i in xrange(n_estimatos):
#                     choice_chunk = choices[i][choices[i] < len(data_chunk)]
#                     data_choice = data_chunk[choice_chunk,:]
#                 
#                     counters[i] += len(data_chunk)
#                     choice = choice[choice >= counter]
#                                             
#             except:
#                 break
# 
#             if data is None:
#                 data = data_chunk  
#             else:
#                 data = numpy.concatenate((data, data_chunk), axis=0)      
#                         
#         return self._get_patterns_labels(data)    
    
    
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
    
    def get_shapes(self):
        
        if self.patterns == True and self.target == True:
            dsetX = self._reader.get("X")
            dsety = self._reader.get("y")
            return dsetX.shape, dsety.shape
        
        elif self.patterns == True:
            dsetX = self._reader.get("X")
            return dsetX.shape
        
        elif self.target == True:
            dsety = self._reader.get("y")
            return dsety.shape           
        
        raise Exception("Either patterns or target must be True!")
    