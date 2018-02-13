#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import numpy

def ensure_data_types(X, y, numpy_dtype_float):
    
    # ensure floats everywhere (e.g., for split array computations)
    if X.dtype != numpy_dtype_float:
        X = X.astype(numpy_dtype_float)
    if y.dtype != numpy_dtype_float:
        y = y.astype(numpy_dtype_float)    
    
    return X, y  
    
class PickableWoodyRFWrapper(object):
    """
    """
    
    def __init__(self, *args):
        
        self.args = args
        
        self.float_type = args[0]
        
        self._params_swig = self.module.PARAMETERS()
        self._forest_swig = self.module.FOREST()

    @property
    def params(self):
        
        return self._params_swig

    @property
    def forest(self):
        
        return self._forest_swig
                    
    @property
    def module(self):
        
        return self._get_wrapper_module()
    
    def _get_wrapper_module(self):
        
        if self.float_type == "float":
            import wrapper_cpu_float
            return wrapper_cpu_float
        elif self.float_type == "double":
            import wrapper_cpu_double
            return wrapper_cpu_double
        
    def __setstate__(self, state):
        """ Is called when object is unpickled
        """        
        
        self.__dict__.update(state)
        
        self._params_swig = self.module.PARAMETERS()
        self._forest_swig = self.module.FOREST()

        self._get_wrapper_module().restore_forest_from_array_extern(self.params, self.forest, self._aforest)
         
    def __getstate__(self):
        """ Is called when object is pickled
        
        https://docs.python.org/3/library/pickle.html#pickle-state
        
        """

        n_bytes_forest = self._get_wrapper_module().get_num_bytes_forest_extern(self.params, self.forest);
        n_bytes_forest = int((float(n_bytes_forest) / 4.0) + 4)

        aforest = numpy.empty(n_bytes_forest, dtype=numpy.int32)
        self._get_wrapper_module().get_forest_as_array_extern(self.params, self.forest, aforest)
        
        state = self.__dict__.copy()
        state['_aforest'] = aforest
        
        del state['_params_swig']
        del state['_forest_swig']
        
        return state        
        