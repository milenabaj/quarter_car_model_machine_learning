"""
Various useful utils.

@author: Milena Bajic (DTU Compute)
"""

import sys,os,logging
import pickle

def get_logger(loger_name, file_handler = None,  formatter = None):
    dlog = logging.getLogger('Datasets')
    dlog.setLevel(logging.DEBUG)
    if file_handler:
        for h in list(dlog.handlers):
            dlog.removeHandler(h)
        if formatter:
            file_handler.setFormatter(formatter)
        dlog.addHandler(file_handler)
    return dlog


def load_pickle(input_dir, string, use_cols = None, row_min = 0, row_max = -1):
    filename = '{0}/{1}'.format(input_dir, string)
    print('Loading: {0}'.format(filename))
    with open(filename, "rb") as f:
        df = pickle.load(f)
        if use_cols:
            return df[use_cols].iloc[row_min:row_max]
        else:
            return df.iloc[row_min:row_max]