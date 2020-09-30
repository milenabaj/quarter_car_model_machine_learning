
"""
Main script.

@author: Milena Bajic (DTU Compute)
"""

import sys,os, glob, time
import logging
from copy import deepcopy
import argparse
import subprocess
from multiprocessing import cpu_count
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from onnx import onnx
from utils import data_loaders
from machine_learning_modules import encoder_decoder, encoder_decoder_with_attention, encoder_decoder_with_speed
from utils import various_utils, plot_utils, model_helpers

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(os.getcwd()) 
sys.path.append(os.getenv("HOME"))
sys.path.append('/home/mibaj/') 
  
def plot_all(a,s):
    plt.figure(figsize=(5,5))
    #plt.ylim((0,1))
    plt.rc('font', size=20)
    plt.plot(a, label = 'Acceleration')
    plt.plot(s, label = 'Severity')
    plt.legend()

    # https://blog.floydhub.com/attention-mechanism/
    
    # Dot attention
    dot = s*a
    dot_t = torch.FloatTensor(dot)
    dot_attn = torch.softmax(dot_t,dim=0)
    cat = np.tanh(s+a)
    concat_attn = torch.softmax(torch.FloatTensor(cat),dim=0)
    
    plt.figure(figsize=(5,5))
    plt.ylim((0.007,0.008))
    plt.title('Attention scores')
    plt.plot(dot_attn, label='Dot attention')
    plt.plot(concat_attn, label='Concat attention')
    plt.legend()
  
    
#logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
if __name__ == "__main__":

    # === SETTINGS === #
    # ================ #
    git_repo_path = subprocess.check_output('git rev-parse --show-toplevel', shell=True, encoding = 'utf-8').strip() 
    
    # Script arguments
    parser = argparse.ArgumentParser(description='Please provide command line arguments.')
    
    # Data preparation
    parser.add_argument('--max_length', default = None,
                        help = 'Max length of sequences in train datasets. If None, it will be computed from the datasets. This variable is used for padding.')  
    parser.add_argument('--speed_min', default = 40, type=int,
                        help = 'Filter datasets based on speed. Pass None for no selection.') 
    parser.add_argument('--speed_max', default = 40.5, type=int,
                        help = 'Filter datasets based on speed. Pass None for no selection.') 
    parser.add_argument('--nrows_to_load', default = 100,
                        help = 'Nrows to load from input (use for testing purposes).')
    
    
    # Training and prediction
    parser.add_argument('--do_train', default = True,
                        help = 'Train using the train dataset.')
    parser.add_argument('--model_type', default = 'lstm_encdec_with_attn',
                        help = 'Choose between lstm_encdec(acceleration sequence -> severity sequence), lstm_encdec_with_attn (with attention) and lstm_encdec_with_speed(acceleration sequence + speed -> severity sequence).')
    parser.add_argument('--do_train_with_early_stopping', default = True,
                        help = 'Do early stopping using the valid dataset (train flag will be set to true by default).')
    parser.add_argument('--do_test', action='store_true',
                        help = 'Test on test dataset.')
    parser.add_argument('--window_size', default = 5, type=int,
                        help = 'Window size.') 

    # Directories
    parser.add_argument('--input_dir', default = '{0}/data/Golden-car-simulation-August-2020/train-val-test-normalized-split-into-windows-size-5'.format(git_repo_path),
                        help = 'Input directory containing train/valid/test subdirectories with prepared data split into windows.')
    parser.add_argument('--out_dir_base', default = 'results',
                        help='Output directory for trained models and results.')
    
    # Run on cluster
    parser.add_argument('--run_on_cluster', action='store_true', 
                        help='Overwrite settings for running on cluster.')
 
    # Parse arguments
    args = parser.parse_args()
    if args.model_type not in ['lstm_encdec', 'lstm_encdec_with_attn', 'lstm_encdec_with_speed']:
        sys.exit('Unknown model passed. Choose beteen lstm_encdec, lstm_encdec_with_attn and lstm_encdec_with_speed.')  
    model_type = args.model_type
    max_length = args.max_length
    speed_selection_range = [args.speed_min, args.speed_max] # Use only data with speed in the selected range
    do_train = args.do_train
    do_train_with_early_stopping = args.do_train_with_early_stopping
    do_test = args.do_test
    window_size = args.window_size #   IMPORTANT 
    nrows_to_load = args.nrows_to_load
    input_dir = args.input_dir
    out_dir_base = args.out_dir_base
    run_on_cluster = args.run_on_cluster # s

    # Other settings
    model_name = model_helpers.get_model_name(model_type)
    acc_to_severity_seq2seq = True # pass True for ac->severity seq2seq or False to do acc->class 
    batch_size = 24
    num_workers = 0 #0
    n_epochs = 1
    learning_rate= 0.001
    patience = 20
    n_pred_plots = 5
    save_results = True
    defect_height_selection = [-200,200]
    defect_width_selection = [0,300]
        
    # ======== SET ========= #
    # ======================= #
    # If run on cluster
    if run_on_cluster:
        input_dir = '/dtu-compute/mibaj/Golden-car-simulation-August-2020/train-val-test-normalized-split-into-windows-size-{0}'.format(window_size)
        out_dir_base = '/dtu-compute/mibaj/Golden-car-simulation-August-2020/results' #a new directory will result will be create here
        nrows_to_load = 10000
        defect_height_selection = [0,5]
        defect_width_selection = [0,10]
        batch_size = 512
        do_test = False
        n_epochs = 50
        n_pred_plots = 100

    # Set flags
    if do_train_with_early_stopping: 
        do_train=True
        
    # Name output directory    
    if args.speed_min and args.speed_max:
        out_dir = '{0}/windowsize_{1}_speedrange_{2}_{3}_{4}_{5}'.format(out_dir_base, window_size, speed_selection_range[0], speed_selection_range[1], model_name, device)
    else:
        out_dir = '{0}/windowsize_{1}_{2}_{3}'.format(out_dir_base, window_size, model_name, device)
    out_dir = '{0}_narrow2020_bid_genattn_teacherforcing_off'.format(out_dir)
    
    # Create output directory      
    if not os.path.exists(out_dir_base):
        os.makedirs(out_dir_base)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # Max length set
    if window_size==5 and args.speed_min==40:
        max_length=136
        
    # Logger
    log = various_utils.get_main_logger('Main', log_filename = 'info.log', log_file_dir = out_dir)
    log.info('======= SETUP =======')
    log.info('Input dir is: {0}'.format(input_dir))
    log.info('Output dir is: {0}'.format(out_dir))
    log.info('Device: {0}'.format(device))
    log.info('Model type: {0}'.format(model_type))
    log.info('Window size: {0}'.format(window_size))
    log.info('Speed selection: {0}'.format(speed_selection_range))
    log.info('Defect width selection: {0}'.format(defect_width_selection)) 
    log.info('Defect height selection: {0}'.format(defect_height_selection)) 
    log.info('====================\n')
    # 7650024 train
    
    
    # ==== PREPARING DATA === #
    # ======================= #
    log.info('Starting preparing the data.\n')
       
    # Train data
    if do_train:
        train_dataset, train_dataloader, max_length =  data_loaders.get_prepared_data(input_dir, out_dir, 'train', acc_to_severity_seq2seq, batch_size, num_workers = num_workers, 
                                                                           max_length = max_length, speed_selection_range =  speed_selection_range, nrows_to_load = nrows_to_load, defect_height_selection = defect_height_selection, defect_width_selection = defect_width_selection)
        

    
  
    df=train_dataset.datasets[0].df
        
    # Pothole
    a = df.acceleration[95] #    defect_height_selection = [-200,200], defect_width_selection = [0,300]
    s=df.severity[95]
    plot_all(a,s)
        
    # if severity is negative -> minimum attention at the most important part -> reverse the sign in drops
    # solution-> make holes dips in att. scores