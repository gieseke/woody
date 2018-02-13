#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import h5py
import numpy
from woody.util import ensure_dir_for_file

class Store(object):
    
    def __init__(self):
        pass

class MemoryStore(Store):
    
    def __init__(self):
        
        self._containers = {}
        self._objects = {}
            
    def create_dataset(self, container_key, dkey, data):
        
        if container_key not in self._containers.keys():
            self._containers[container_key] = {}
        
        self._containers[container_key][dkey] = data    
            
    def append_to_dataset(self, container_key, dkey, data):

        if container_key not in self._containers.keys():
            self._containers[container_key] = {}
                    
        if not dkey in self._containers[container_key].keys():
            
            self._containers[container_key][dkey] = data
            
        else:
            
            newdata = numpy.concatenate([self._containers[container_key][dkey], data], axis=0)
            self._containers[container_key][dkey] = newdata 
        
    def get_dataset(self, container_key, dkey):
        
        return numpy.ascontiguousarray(self._containers[container_key][dkey])
    
    def get_keys(self, container_key):
        
        return self._containers[container_key].keys()        
    
    def save(self, key, obj):
        
        self._objects[key] = obj
    
    def load(self, key, loader):
        
        return self._objects[key]
    
class DiskStore(Store):
    
    def __init__(self):
        pass 
            
    def create_dataset(self, container_key, dkey, data):
        
        ensure_dir_for_file(container_key)  
        s = h5py.File(container_key, 'a', driver="sec2", libver='latest')
        
        dset = s.create_dataset(dkey, data.shape, maxshape=(None, data.shape[1]), compression="lzf")
        dset[:,:] = data
        
        s.close()
    
    def append_to_dataset(self, container_key, dkey, data):

        ensure_dir_for_file(container_key)
        s = h5py.File(container_key, 'a', driver="sec2", libver='latest')
        
        offset = 0
        
        if not dkey in s.keys():
            
            dset = s.create_dataset(dkey, data.shape, maxshape=(None, data.shape[1]), compression="lzf")
            
        else:
            
            dset = s.get(dkey)
            offset += dset.shape[0]
            dset.resize(dset.shape[0] + data.shape[0], axis=0)
    
        dset[offset:, :] = data
        
        s.close()
        
    def get_dataset(self, container_key, dkey):
        
        with h5py.File(container_key, 'r') as container:
            dset = numpy.array(container.get(dkey))
            
        return dset[:,:]
    
    def get_keys(self, container_key):
                
        s = h5py.File(container_key, 'r')
        keys = s.keys()
        s.close()
        
        return keys  
    
    def save(self, key, obj):
        
        obj.save(key)
    
    def load(self, key, loader):
        
        return loader.load(key)
