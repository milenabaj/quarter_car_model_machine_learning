"""
PyTorch Data loading utils.

@author: Milena Bajic (DTU Compute)
"""

import sys,os, glob, time
import subprocess
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from quarter_car_model_machine_learning.utils.various_utils import *

# Get logger for module
dlog = get_mogule_logger("data_loaders")

# get root logger by logging.getLogger() and its handler
class Dataset(Dataset):
    def __init__(self, filename, filetype, mode, max_length = None):

        # Take input
        self.filename = filename
        self.filename_bare = filename.split('/')[-1]
        self.filetype = filetype
        self.mode = mode
        if max_length:
            self.max_length = max_length

        # Load features and targets
        self.load_data()

        # Number of samples
        self.n_samples = self.acc.shape[0]

    def load_data(self):
        dlog.info('Loading: {0}\n'.format(self.filename))

        file = load_pickle_full_path(self.filename)
        self.acc = file.acceleration.to_numpy()
        if not self.max_length:
            self.get_max_length()
        self.acc = self.pad_arrays(self.acc)
        if self.mode == 'acc-severity':
            self.severity = file.severity.to_numpy()
            self.severity = self.pad_arrays(self.severity)
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


def get_datasets(input_dir, filetype, mode, batch_size = 'full_dataset', num_workers = 0):
    '''
    Get a list of (filename, Dataset, Dataloader) for each file in directory.
    '''
    '''
    print('Finding max length')
    glob_max_length = 0
    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):
        file = load_pickle_full_path(filename)
        acc = file.acceleration.to_numpy()
        lengths = [len(s) for s in acc]
        max_length = max(lengths)
        if max_length>glob_max_length:
            glob_max_length = max_length
    print('Max length: ', max_length)
    '''
    max_length = 2001 # change this (uncomment upper part) for new data
    data = []
    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):

        dataset =  Dataset(filename=filename, filetype = filetype, mode = mode, max_length=max_length)
        if batch_size=='full_dataset':
            batch_size = dataset.n_samples

        dataloader = DataLoader(dataset, batch_size = batch_size, num_workers=num_workers)
        #data.append((filename.split('/')[-1], dataset,dataloader))
        data.append(dataset)

    return data

def get_concat_data(input_dir, filetype, mode, batch_size = 'full_dataset', num_workers = 0):
    '''
    Get a list of (filename, Dataset, Dataloader) for each file in directory.
    '''
    '''
    print('Finding max length')
    glob_max_length = 0
    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):
        file = load_pickle_full_path(filename)
        acc = file.acceleration.to_numpy()
        lengths = [len(s) for s in acc]
        max_length = max(lengths)
        if max_length>glob_max_length:
            glob_max_length = max_length
    print('Max length: ', max_length)
    '''
    max_length = 2001 # change this (uncomment upper part) for new data
    data = []
    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):

        dataset =  Dataset(filename=filename, filetype = filetype, mode = mode, max_length=max_length)
        if batch_size=='full_dataset':
            batch_size = dataset.n_samples

        dataloader = DataLoader(dataset, batch_size = batch_size, num_workers=num_workers)
        #data.append((filename.split('/')[-1], dataset,dataloader))
        data.append(dataset)

    return data