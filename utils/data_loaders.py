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
sys.path.append(os.getcwd()) 
sys.path.append(os.getenv("HOME"))
sys.path.append('/home/mibaj/') 
   
from quarter_car_model_machine_learning.utils.various_utils import *

# Get logger for module
dlog = get_mogule_logger("data_loaders")

   
def get_dataset_max_length(input_dir, filetype, num_workers = 0, speed_selection_range = None, nrows_to_load = -1): 
    '''
    Get max length of sequences in input datasets.
    '''
    dlog.info('===> Getting max lenght for datasets in: {0}'.format(input_dir))
    glob_max_length = 0
    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):
        print('filename',filename)
        file = load_pickle_full_path(filename, speed_selection_range = speed_selection_range, row_max = nrows_to_load)
        
        # This file max lenght
        orig_lengths = file.acceleration.apply(lambda row: row.shape[0]).to_numpy(dtype='int')
        #print('orig_lengths', orig_lengths[0])
        this_file_max_length = np.max(orig_lengths)
        
        if this_file_max_length>glob_max_length:
            glob_max_length = this_file_max_length
    dlog.info('Max length is: {0}\n'.format(glob_max_length))
    return glob_max_length
  
    
# Functions used for getting the data #
# =================================== #    
def get_prepared_data(input_dir, filetype, acc_to_severity_seq2seq, batch_size, num_workers = 0, max_length = None, speed_selection_range = None, nrows_to_load = -1):
    datasets = get_datasets(input_dir, filetype, acc_to_severity_seq2seq, num_workers = num_workers, max_length =  max_length,  speed_selection_range = speed_selection_range, nrows_to_load = nrows_to_load) 
    merged_dataset = ConcatDataset(datasets)
    merged_dataloader = DataLoader(merged_dataset, batch_size = batch_size, num_workers=num_workers)
    return datasets, merged_dataloader


def get_datasets(input_dir, filetype, acc_to_severity_seq2seq, num_workers = 0, max_length = None, speed_selection_range = None, nrows_to_load = -1):
    '''
    Get a list of (filename, Dataset, Dataloader) for each file in directory.
    '''
    dlog.info('\n===> Getting datasets for filetype: {0}'.format(filetype))
    data = []
    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):
        dataset = Dataset(filename=filename, filetype = filetype,acc_to_severity_seq2seq = acc_to_severity_seq2seq, max_length=max_length, speed_selection_range = speed_selection_range, nrows_to_load = nrows_to_load)
        data.append(dataset)
    dlog.info('\n')
    return data


class Dataset(Dataset):
    def __init__(self, filename, filetype, acc_to_severity_seq2seq, max_length,  speed_selection_range = None, nrows_to_load = -1):
        dlog.debug('=> Creating dataset for file {0}'.format(filename))
        
        # Take input
        self.filename = filename
        self.filename_bare = filename.split('/')[-1]
        self.filetype = filetype
        self.acc_to_severity_seq2seq = acc_to_severity_seq2seq
        self.speed_selection_range = speed_selection_range
        self.nrows_to_load  = nrows_to_load
        self.max_length = max_length

        # Load features and targets
        self.load_data()

        # Number of samples
        self.n_samples = self.acc.shape[0]

    def load_data(self):
        dlog.info('Loading: {0}'.format(self.filename))
        file = load_pickle_full_path(self.filename, speed_selection_range = self.speed_selection_range, row_max = self.nrows_to_load)
        self.df = file
        
        # Save original lengths
        self.orig_length = self.df.acceleration.apply(lambda row: row.shape[0]).to_numpy(dtype='int')
        
        # Get and pad inputs
        self.acc = self.pad_arrays(file.acceleration)
        self.speed = file.speed.to_numpy(dtype='float32')
        
        # Get and pad targets
        if self.acc_to_severity_seq2seq:
            self.severity = self.pad_arrays(file.severity)
        else:
            self.window_class = file.window_class.to_numpy() # does not need padding
        return

    def pad_arrays(self, arrays):
        arrays = arrays.to_numpy()
        padded_list = [ np.pad(arr,  (0,(self.max_length-arr.shape[0])),  mode='constant') for arr in arrays ] # padded at the end of each array
        return np.array(padded_list, dtype=np.float32)
        # padding helps classification, as it stores signal lenght (like velocity)

    def __getitem__(self, index):

        if self.acc_to_severity_seq2seq:
            return self.acc[index], self.speed[index], self.orig_length[index], self.severity[index]

        else:
            return self.acc[index], self.speed[index], self.orig_length[index], self.window_class[index]

    def __len__(self):
        return self.n_samples

# =================================== # 

