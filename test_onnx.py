"""
Onnx test

@author: Milena Bajic (DTU Compute)
"""

import sys,os, glob, time
import subprocess
import argparse
import pickle
from onnx import onnx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils import data_loaders
from machine_learning_modules import encoder_decoder
from utils import various_utils, plot_utils, model_helpers
from onnx import onnx
import onnxruntime
from machine_learning_modules import encoder_decoder
from utils import various_utils, plot_utils, model_helpers

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === SETTINGS === #
# ================ #
git_repo_path = subprocess.check_output('git rev-parse --show-toplevel', shell=True, encoding = 'utf-8').strip() 

# Script arguments
parser = argparse.ArgumentParser(description='Please provide command line arguments.')

# Data preparation
parser.add_argument('--max_length', default = None,
                    help = 'Max length of sequences in train datasets. If None, it will be computed from the datasets. This variable is used for padding.')  
parser.add_argument('--speed_selection_range', default = [60,80], 
                    help = 'Select datasets for this speed only. Pass None for no selection.') 
parser.add_argument('--nrows_to_load', default = 100,
                    help = 'Nrows to load from input (use for testing purposes). If speed selelection range is not None, all rows will be used.')


# Training and prediction
parser.add_argument('--do_train', default = True,
                    help = 'Train using the train dataset.')
parser.add_argument('--do_train_with_early_stopping', default = True,
                    help = 'Do early stopping using the valid dataset (train flag will be set to true by default).')

# Directories
parser.add_argument('--input_dir', default = '{0}/data/Golden-car-simulation-August-2020/train-val-test-normalized-split-into-windows-cluster'.format(git_repo_path),
                    help = 'Input directory containing train/valid/test subdirectories with prepared data split into windows.')
parser.add_argument('--output_dir', default = 'output',
                    help='Output directory for trained models and results.')

# Parse arguments
args = parser.parse_args()
max_length = args.max_length
speed_selection_range = args.speed_selection_range 
do_train = args.do_train
do_train_with_early_stopping = args.do_train_with_early_stopping
nrows_to_load = args.nrows_to_load
if do_train_with_early_stopping: 
    do_train=True

# Other settings
save_results = True
acc_to_severity_seq2seq = True # pass True for ac->severity seq2seq or False to do acc->class 
model_name = 'LSTM_encoder_decoder'
batch_size = 50 #'full_dataset'
num_workers = 0 #0
n_epochs = 1
learning_rate= 0.001
patience = 30

# Input and output directory
input_dir = args.input_dir
if speed_selection_range:
    out_dir = '{0}_{1}_speedrange_{2}_{3}_{4}'.format(args.output_dir, model_name, speed_selection_range[0], speed_selection_range[1], device)
else:
    out_dir = '{0}_{1}_{2}'.format(args.output_dir, model_name, device)

# Logger
log = various_utils.get_main_logger('Main', log_filename = 'info.log', log_file_dir = out_dir)
log.info('Output dir is: {0}/{1}\n'.format(os.getcwd(), out_dir))        

if not max_length:
    max_length = data_loaders.get_dataset_max_length(input_dir, 'train', num_workers = 0,  speed_selection_range =  speed_selection_range, 
                                                          nrows_to_load = nrows_to_load)
# Train data, # change max_length to be computed
train_datasets, train_dataloader =  data_loaders.get_prepared_data(input_dir, 'train', acc_to_severity_seq2seq, batch_size, num_workers = num_workers, 
                                                                       max_length = max_length, speed_selection_range =  speed_selection_range,  
                                                                       nrows_to_load = nrows_to_load)

# Valid data
valid_datasets, valid_dataloader =  data_loaders.get_prepared_data(input_dir, 'valid', acc_to_severity_seq2seq, batch_size, num_workers = num_workers, 
                                                                       max_length = max_length,  speed_selection_range =  speed_selection_range,
                                                                       nrows_to_load = nrows_to_load)

 
for batch_index, (features, targets) in enumerate(train_dataloader):
    log.debug('Batch_index: {0}'.format(batch_index))

    # Put into the correct dimensions for LSTM
    features = features.permute(1,0)
    features = features.unsqueeze(2).to(device)

    targets = targets.permute(1,0)
    targets = targets.unsqueeze(2).to(device)
                        
# Test onnx
onnx_path = 'output_LSTM_encoder_decoder_cpu/trained_model_LSTM_encoder_decoder.onnx'

# Onnx input
onnx_input = (features, features)

# Run
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
ses = onnxruntime.InferenceSession(onnx_path)
input_name = ses.get_inputs()[0].name
output_name = ses.get_outputs()[0].name

x = np.random.random((34,50,1))
x = x.astype(np.float32)
res = ses.run([output_name], {input_name: x})
pred = res[0].shape
