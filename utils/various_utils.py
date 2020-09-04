"""
Various useful utils.

@author: Milena Bajic (DTU Compute)
"""

import sys,os,logging
import pickle

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
        
    # Create stream handler
    ch = logging.StreamHandler()
    log_ed.addHandler(ch)

    # Get the file handler from the root logger and add it to the module logger
    for h in list(root_logger.handlers):
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


def load_pickle(input_dir, string, use_cols = None, row_min = 0, row_max = -1):
    filename = '{0}/{1}'.format(input_dir, string)
    with open(filename, "rb") as f:
        df = pickle.load(f)
        if use_cols:
            return df[use_cols].iloc[row_min:row_max]
        else:
            return df.iloc[row_min:row_max]

def load_pickle_full_path(filename, use_cols = None, row_min = 0, row_max = -1):
    with open(filename, "rb") as f:
        df = pickle.load(f)
        if use_cols:
            return df[use_cols].iloc[row_min:row_max]
        else:
            return df.iloc[row_min:row_max]