"""
Various useful utils.

@author: Milena Bajic (DTU Compute)
"""

import sys,os,logging
import numpy as np
from copy import deepcopy
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def get_mogule_logger(logger_name, root_logger_name='Main'):
    import logging
    
    # Create module logger
    log_ed = logging.getLogger(logger_name)

    # Set the same level as in the root logger
    root_logger = logging.getLogger(root_logger_name)
    log_ed.setLevel(root_logger.getEffectiveLevel())
    
    # Remove possible handlers
    for h in list(log_ed.handlers):
        log_ed.removeHandler(h)
        
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    
    # Create stream handler
    ch_ed = logging.StreamHandler()
    ch_ed.setFormatter(formatter)
    log_ed.addHandler(ch_ed)

    # Get the file handler from the root logger and add it to the module logger
    for h in list(root_logger.handlers):
        if isinstance(h,logging.FileHandler):
            log_ed.addHandler(h)
        
    return log_ed

def get_main_logger(logger_name = 'Main', log_filename = 'info.log', log_level = logging.DEBUG, 
                    log_file_dir = '.'):
    import logging
    
    #Create logger
    log = logging.getLogger(logger_name)
    log.setLevel(log_level)
    for h in list(log.handlers):
        log.removeHandler(h)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # Create stream handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    log.addHandler(ch)
    
    # Create file handler
    log_filename = '{0}/{1}'.format(log_file_dir, log_filename)
    fh = logging.FileHandler(log_filename, mode='w')
    fh.setFormatter(formatter)
    log.addHandler(fh)
    
    return log



def load_pickle_full_path(filename, use_cols = None, speed_selection_range = None, row_min = 0, row_max = -1):
    log_vu.info('Loading {0} rows. Speed selection range: {1}'.format(row_max, speed_selection_range))
    with open(filename, "rb") as f:
        df = pickle.load(f)
        if speed_selection_range:
            speed_min = speed_selection_range[0]
            speed_max = speed_selection_range[1]
            cond = (df.speed > speed_min) & (df.speed < speed_max)
            df = df[cond]
            df.reset_index(inplace=True, drop=True)
        if use_cols:
            df =  df[use_cols].iloc[row_min:row_max]
        else:
            df = df.iloc[row_min:row_max]
        return df
      
        
                   
    
# Get logger for module
log_vu = get_mogule_logger("various_utils")