#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import time

class Timer(object):
    
    def __init__(self):
        
        self._start_time = 0.0
        self._elapsed_time = 0.0
    
    def start(self):
        
        self._start_time = time.time()
        
    def stop(self):
        
        self._elapsed_time += time.time() - self._start_time
        self._start_time = 0.0
    
    def reset(self):
        
        self._start_time = 0.0
        self._elapsed_time = 0.0
        
    def get_elapsed_time(self):
        
        return self._elapsed_time
        