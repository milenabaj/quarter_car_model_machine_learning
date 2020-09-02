"""
Various useful utils.

@author: Milena Bajic (DTU Compute)
"""

import sys,os,logging


def get_logger(loger_name, file_handler = None):
    dlog = logging.getLogger('Datasets')
    dlog.setLevel(logging.DEBUG)
    for h in list(dlog.handlers):
        dlog.removeHandler(h)
    dlog.addHandler(file_handler)
    return dlog