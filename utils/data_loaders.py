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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from quarter_car_model_machine_learning.utils.various_utils import *

# Get logger for module
dlog = get_mogule_logger("data_loaders")

   
def get_dataset_max_length(input_dir, filetype, num_workers = 0, nrows_to_load = -1): 
    '''
    Get max length of sequences in input datasets.
    '''
    dlog.info('===> Getting max lenght for datasets in: {0}'.format(input_dir))
    glob_max_length = 0
    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):
        file = load_pickle_full_path(filename, row_max = nrows_to_load)
        acc = file.acceleration.to_numpy()
        lengths = [len(s) for s in acc]
        max_length = max(lengths)
        if max_length>glob_max_length:
            glob_max_length = max_length
    dlog.info('Max length is: {0}\n'.format(glob_max_length))
    return glob_max_length
  
    
def get_prepared_data(input_dir, filetype, mode, batch_size, num_workers = 0, max_length = None, nrows_to_load = -1):
    datasets = get_datasets(input_dir, filetype, mode, num_workers = num_workers, max_length =  max_length, nrows_to_load = nrows_to_load) 
    merged_dataset = ConcatDataset(datasets)
    merged_dataloader = DataLoader(merged_dataset, batch_size = batch_size, num_workers=num_workers)
    return datasets, merged_dataloader


def get_datasets(input_dir, filetype, mode, num_workers = 0, max_length = None, nrows_to_load = -1):
    '''
    Get a list of (filename, Dataset, Dataloader) for each file in directory.
    '''
    dlog.info('\n===> Getting datasets for filetype: {0}'.format(filetype))
    data = []
    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):
        dataset = Dataset(filename=filename, filetype = filetype, mode = mode, max_length=max_length, nrows_to_load = nrows_to_load)
        data.append(dataset)
    dlog.info('\n')
    return data


class Dataset(Dataset):
    def __init__(self, filename, filetype, mode, max_length = None, nrows_to_load = -1):
        dlog.debug('=> Creating dataset for file {0}'.format(filename))
        
        # Take input
        self.filename = filename
        self.filename_bare = filename.split('/')[-1]
        self.filetype = filetype
        self.mode = mode
        self.nrows_to_load  = nrows_to_load
        if max_length:
            self.max_length = max_length

        # Load features and targets
        self.load_data()

        # Number of samples
        self.n_samples = self.acc.shape[0]

    def load_data(self):
        dlog.info('Loading: {0}'.format(self.filename))
        file = load_pickle_full_path(self.filename, row_max = self.nrows_to_load)
        self.acc = file.acceleration.to_numpy()
        if not self.max_length:
            self.get_max_length()
            
        # Get and pad features
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



