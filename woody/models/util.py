#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import logging
from datetime import datetime

from logging.handlers import RotatingFileHandler

def init_logger(fname, log_name="Logger", log_level="INFO"):
    
    # create logging directory if needed
    d = os.path.dirname(fname)
    if not os.path.exists(d):
        os.makedirs(d)
            
    logger = logging.getLogger(log_name + "_" + str(datetime.now()))
    if log_level == 'INFO': 
        logger.setLevel(logging.INFO)
    else: 
        logger.setLevel(logging.DEBUG)
        
    # logging formatter
    #formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # store output if specified
    if fname is not None:
        log_handler = RotatingFileHandler(fname, 'a')
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)
    
    # standard streaming handler 
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # avoid double outputs
    logger.propagate = 0

    return logger 