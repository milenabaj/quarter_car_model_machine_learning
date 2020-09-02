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
from quarter_car_model_machine_learning.utils.various_utils import *

# TRY MULTIPLE DATALOADERS FOR EACH FILE
# load file inside of getindex

class Dataset(Dataset):
    def __init__(self, input_dir_base, filetype, batch_size = 1, file_handler = None,  formatter = None):

        # Take input from user
        self.input_dir = '{0}/{1}'.format(input_dir_base,filetype)
        self.filetype = filetype
        self.data_files = glob.glob('{0}/*.pkl'.format(self.input_dir))
        self.data_files_names_only = [f.split('/')[-1] for f in self.data_files]
        self.batch_size = batch_size
        self.use_cols = ['acceleration', 'window_class', 'speed', 'defect_width', 'defect_height']
        self.length = 0
        #for file in self.data_files_names_only:
        #    nrows = load_pickle(self.input_dir, file, use_cols = self.use_cols).shape[0]
        #    print('nrows: ', nrows)
        #    self.length = self.length + nrows

        #Create logger
        self.dlog = get_logger('Datasets', file_handler, formatter)
        self.dlog.info('List of input data files: {0}\n'.format(self.data_files_names_only))

    def load_file(self, num):
        return load_pickle(self.input_dir, self.data_files_names_only[num], use_cols = self.use_cols)

    def __getindex__(self, idx):
        load_pickle(self.input_dir, self.data_files_names_only[num], use_cols = self.use_cols)
        return
        #return load_file(self.data_files[idx])

    def __len__(self):
        return self.length

    # for batch_idx 0: load 1st file, batch size lines
    # for batch_idx 1: load 1st file, next batch size lines
    # define dataset and dataloader outside
    # dataset needs to remember which file and line was called in the last call

