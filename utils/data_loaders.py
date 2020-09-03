"""
PyTorch Data loading utils.

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

        # Load features and targets
        self.load_data()

        # Number of samples
        self.n_samples = self.acc.shape[0]

    def load_data(self):
        self.dlog.info('Loading: {0}\n'.format(self.filename))

        file = load_pickle_full_path(self.filename)
        self.acc = file.acceleration.to_numpy()
        self.get_max_length()
        self.padded_acc = self.pad_arrays(self.acc)
        if self.mode == 'acc-severity':
            self.severity = file.severity.to_numpy()
            self.severity  = self.pad_arrays(self.severity)
        elif self.mode == 'classification':
            self.window_class = file.window_class.to_numpy() # does not need padding
        elif self.mode == 'exploration':
            self.file = file
        return

    def get_max_length(self):
        lengths = [len(s) for s in self.acc]
        self.max_length = max(lengths)
        self.max_length_index = lengths.index(self.max_length )
        return

    def pad_arrays(self, arrays):
        padded_list = [ np.pad(arr,  (0,(self.max_length-arr.shape[0])),  mode='constant') for arr in arrays ]
        return np.array(padded_list, dtype=np.float32)

    def __getitem__(self, index):

        if self.mode == 'acc-severity':
            return self.acc[index], self.severity[index]

        elif self.mode == 'classification':
            return self.acc[index], self.window_class[index]

    def __len__(self):
        return self.n_samples


def get_datasets(input_dir, filetype, mode, batch_size = 'full_dataset', num_workers = 0, file_handler = None, formatter = None):
    '''
    Get a list of (filename, Dataset, Dataloader) for each file in directory.
    '''
    print('Finding max length')
    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):
        file = load_pickle_full_path(self.filename)
        acc = file.acceleration.to_numpy()
        lengths = [len(s) for s in self.acc]
        self.max_length = max(lengths)
        self.max_length_index = lengths.index(self.max_length )

    data = []
    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):

        dataset =  Dataset(filename=filename, filetype = filetype, mode = mode,
                                              file_handler = file_handler, formatter = formatter)
        if batch_size=='full_dataset':
            batch_size = dataset.n_samples

        dataloader = DataLoader(dataset, batch_size = batch_size, num_workers=num_workers)
        data.append((filename.split('/')[-1], dataset,dataloader))
    return data