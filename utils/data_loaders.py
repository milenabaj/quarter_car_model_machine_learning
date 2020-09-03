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
from torch.utils.data import Dataset, DataLoader, IterableDataset
from quarter_car_model_machine_learning.utils.various_utils import *

# TRY MULTIPLE DATALOADERS FOR EACH FILE
# load file inside of getindex


'''
class Dataset(IterableDataset):
    # needs to be used with n_workers=0 in dataloader
    def __init__(self, input_dir_base, filetype, batch_size = 1, file_handler = None,  formatter = None):

        # Take input from user
        self.input_dir = '{0}/{1}'.format(input_dir_base,filetype)
        self.filetype = filetype
        self.data_files = glob.glob('{0}/*.pkl'.format(self.input_dir))
        self.data_files_names_only = [f.split('/')[-1] for f in self.data_files]
        self.batch_size = batch_size
        #self.use_cols = ['acceleration', 'window_class', 'speed', 'defect_width', 'defect_height']
        self.use_cols = ['acceleration', 'severity']
        self.length = 0

        #Create logger
        self.dlog = get_logger('Datasets', file_handler, formatter)
        self.dlog.info('List of input data files: {0}\n'.format(self.data_files_names_only))

    def load_file(self, num):
        return load_pickle(self.input_dir, self.data_files_names_only[num], use_cols = self.use_cols, row_max=10)

    def __iter__(self):

         #Create an iterator
        print('Loading ',self.data_files_names_only[0])
        f = load_pickle(self.input_dir, self.data_files_names_only[0], use_cols = self.use_cols, row_max=10)
        acc = np.nditer(f.acceleration.to_numpy(),flags=['refs_ok'])
        sev = np.nditer(f.severity.to_numpy(),flags=['refs_ok'])
        #f = load_pickle(self.input_dir, self.data_files_names_only[0], use_cols = self.use_cols, row_max=10).iterrows()

        return acc, sev

'''



class Dataset(Dataset):
    def __init__(self, filename, filetype, mode, file_handler = None,  formatter = None):

        #Create logger
        self.dlog = get_logger('Datasets', file_handler, formatter)

        # Take input
        self.filename = filename
        self.filename_bare = filename.split('/')[-1]
        self.filetype = filetype
        self.mode = mode

        # Mode
        if self.mode == 'acc-severity':
            self.use_cols = ['acceleration', 'severity']
        elif self.mode == 'classification':
            self.use_cols = ['acceleration', 'window_class']
        elif self.mode == 'exploration':
            self.use_cols = ['acceleration', 'window_class', 'speed', 'defect_width', 'defect_height']

        # Load file
        self.file = self.load_file(self.filename)

        # Number of samples
        self.length = self.file.shape[0]

    def load_file(self, filename):

        self.dlog.info('Loading: {0}\n'.format(filename))
        return load_pickle_full_path(filename, use_cols = self.use_cols)

    def __getindex__(self, index):

        if self.mode == 'acc-severity':
            acc = f.acceleration.to_numpy()
            severity = f.severity.to_numpy()
            return acc[index], severity[index]

        elif self.mode == 'classification':
            acc = f.acceleration.to_numpy()
            window_class = f.window_class.to_numpy()
            return acc[index], window_class[index]

    def __len__(self):
        return self.length

    # for batch_idx 0: load 1st file, batch size lines
    # for batch_idx 1: load 1st file, next batch size lines
    # define dataset and dataloader outside
    # dataset needs to remember which file and line was called in the last call

def get_datasets(input_dir, filetype, mode, batch_size = 16, num_workers = 0, file_handler = None, formatter = None):
    data = []
    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):
        dataset =  Dataset(filename=filename, filetype = filetype, mode = mode,
                                              file_handler = file_handler, formatter = formatter)
        dataloader = DataLoader(dataset, batch_size = batch_size, num_workers=num_workers)
        data.append((filename.split('/')[-1], dataset,dataloader))
    return data