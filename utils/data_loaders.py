"""
Data loading utils.

@author: Milena Bajic (DTU Compute)
"""

import sys,os, glob, time, logging
import subprocess
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from quarter_car_model_machine_learning.data_preparation_modules.normalize_data import load_pickle
from quarter_car_model_machine_learning.utils.various_utils import get_logger

class Datasets(Dataset):
    def __init__(self, input_dir, filetype, file_handler = None,  formatter = None):

        # Take input from user
        self.input_dir = '{0}/{1}'.format(input_dir,filetype)
        self.filetype = filetype
        self.data_files = glob.glob('{0}/*.pkl'.format(self.input_dir))

        #Create logger
        self.dlog = get_logger('Datasets', file_handler, formatter)
        self.dlog.info('List of input data files: {0}\n'.format([f.split('/')[-1] for f in self.data_files]))

    def __getindex__(self, idx):
        return
        #return load_file(self.data_files[idx])

    def __len__(self):
        return len(self.data_files)

    # for batch_idx 0: load 1st file, batch size lines
    # for batch_idx 1: load 1st file, next batch size lines
    # define dataset and dataloader outside
    # dataset needs to remember which file and line was called in the last call

