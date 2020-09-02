"""
Various useful utils.

@author: Milena Bajic (DTU Compute)
"""

import sys,os,logging


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