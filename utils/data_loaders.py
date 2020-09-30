"""
PyTorch Data loading utils.

@author: Milena Bajic (DTU Compute)
"""

import sys,os, glob, time
import subprocess
import argparse
import logging
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

   
def get_dataset_max_length(input_dir, filetype, num_workers = 0, speed_selection_range = None, nrows_to_load = -1,
                           defect_height_selection = None, defect_width_selection = None): 
    '''
    Get max length of sequences in input datasets.
    '''
    dlog.info('===> Getting max lenght for datasets in: {0}'.format(input_dir))

    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):
        file = load_pickle_full_path(filename, speed_selection_range = speed_selection_range, row_max = nrows_to_load,
                                     defect_height_selection = defect_height_selection, defect_width_selection  = defect_width_selection )   
        
        if file.empty:
            continue #this selection
        
        # Min speed
        if speed_selection_range:
            min_speed = speed_selection_range[0]
        else:
            min_speed = 2
        
        # Get acc length for min speed
        if file.acceleration[file.speed == min_speed].empty:
            continue
        
        max_length = file.acceleration[file.speed == min_speed].iloc[0].shape[0]
        return max_length
        
  
    
# Functions used for getting the data #
# =================================== #    
def get_prepared_data(input_dir, out_dir, filetype, acc_to_severity_seq2seq, batch_size, num_workers = 0, max_length = None, speed_selection_range = None, nrows_to_load = -1,
                      defect_height_selection = None, defect_width_selection = None): 
        
    # The data will be padded to max_length
    if not max_length and filetype=='train':
         max_length = get_dataset_max_length(input_dir, 'train', num_workers = 0,  speed_selection_range =  speed_selection_range, 
                                                          nrows_to_load = nrows_to_load, defect_height_selection = defect_height_selection,
                                                          defect_width_selection  = defect_width_selection )      
    dlog.info('Max length: {0}'.format(max_length))

    # Filename with the prepared data for this filetype
    out_filename = '{0}/{1}_dataset_speed_{2}_{3}_defwidth_{4}_{5}_defheight_{6}_{7}.pth'.format(out_dir,filetype, speed_selection_range[0], speed_selection_range[1],
                                                                                                  defect_width_selection[0], defect_width_selection[1],defect_height_selection[0],defect_height_selection[1])                                                                                                                                                                                          
    # Load if exists, else create
    if os.path.exists(out_filename):
        merged_dataset = torch.load(out_filename) 
    else:
        datasets = get_datasets(input_dir, filetype, acc_to_severity_seq2seq, num_workers = num_workers, max_length =  max_length,  speed_selection_range = speed_selection_range, 
                                nrows_to_load = nrows_to_load, defect_height_selection = defect_height_selection,  defect_width_selection =  defect_width_selection) 
        merged_dataset = ConcatDataset(datasets)        
        torch.save(merged_dataset, out_filename)
        
    # Create the dataloader 
    merged_dataloader = DataLoader(merged_dataset, batch_size = batch_size, num_workers=num_workers)
    n_samples = sum(merged_dataloader.dataset.cumulative_sizes)
    dlog.info('{0} samples.\n'.format(n_samples))
          
    return merged_dataset, merged_dataloader, max_length


def get_datasets(input_dir, filetype, acc_to_severity_seq2seq, num_workers = 0, max_length = None, speed_selection_range = None, nrows_to_load = -1,
                 defect_height_selection = None, defect_width_selection = None):
    '''
    Get a list of (filename, Dataset, Dataloader) for each file in directory.
    '''
    dlog.info('\n===> Getting datasets for filetype: {0}'.format(filetype))
    data = []
    for filename in glob.glob('{0}/{1}/*.pkl'.format(input_dir, filetype)):
        file = load_pickle_full_path(filename, speed_selection_range = speed_selection_range, row_max = nrows_to_load,
                                     defect_height_selection = defect_height_selection, defect_width_selection = defect_width_selection )
        if file.empty:
            continue #this selection
        dataset = Dataset(filename=filename, filetype = filetype,acc_to_severity_seq2seq = acc_to_severity_seq2seq, max_length=max_length, speed_selection_range = speed_selection_range, nrows_to_load = nrows_to_load)
        data.append(dataset)
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
        self.scaled_speed = file.scaled_speed.to_numpy(dtype='float32')
            
        # Get and pad targets
        if self.acc_to_severity_seq2seq:
            self.severity = self.pad_arrays(file.severity)
        else:
            self.window_class = file.window_class.to_numpy() # does not need padding
        return

    def pad_arrays(self, arrays):
        arrays = arrays.to_numpy()
        padded_list = [ np.pad(arr,  ((self.max_length-arr.shape[0]),0),  mode='constant') for arr in arrays ] # padded at the end of each array
        return np.array(padded_list, dtype=np.float32)
        # padding helps classification, as it stores signal lenght (like velocity)

    def __getitem__(self, index):

        if self.acc_to_severity_seq2seq:
            return self.acc[index], self.scaled_speed[index], self.speed[index], self.orig_length[index], self.severity[index]

        else:
            return self.acc[index], self.scaled_speed[index], self.speed[index], self.orig_length[index], self.window_class[index]

    def __len__(self):
        return self.n_samples

# =================================== # 

