"""
Main script.

@author: Milena Bajic (DTU Compute)
"""

import sys,os, glob, time
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
from utils import data_loaders
from machine_learning_modules import encoder_decoder
from utils import various_utils


if __name__ == "__main__":

    # === SETTINGS === #
    # ================ #
    git_repo_path = subprocess.check_output('git rev-parse --show-toplevel', shell=True, encoding = 'utf-8').strip() 
    
    # Script arguments
    parser = argparse.ArgumentParser(description='Please provide command line arguments.')
    parser.add_argument('--nrows_to_load', default = 100,
                        help = 'Nrows to load from input (use for testing purposes).')
    parser.add_argument('--do_train', default = True,
                        help = 'Use train dataset to train.')
    parser.add_argument('--do_train_with_early_stopping', default = True,
                        help = 'Use train dataset to train and valid to do early stopping.')
    parser.add_argument('--input_dir', default = '{0}/data/Golden-car-simulation-August-2020/train-val-test-normalized-split-into-windows-cluster'.format(git_repo_path),
                        help = 'Input directory containing train/valid/test subdirectories with prepared data split into windows.')
    parser.add_argument('--output_dir', default = 'output',
                        help='Output directory for trained models and results.')

    # Parse arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    out_dir = args.output_dir
    nrows_to_load = args.nrows_to_load
    do_train = args.do_train
    do_train_with_early_stopping = args.do_train_with_early_stopping
    if do_train_with_early_stopping: 
        do_train=True
        
    # Other settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = 'acc-severity'
    batch_size = 20 #'full_dataset'
    num_workers = 0 #0
    n_epochs = 1
    learning_rate= 0.001

    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Logger
    log = various_utils.get_main_logger('Main', log_filename = 'info.log', log_file_dir = out_dir)
    log.info('Output dir is: {0}/{1}\n'.format(os.getcwd(), out_dir))

    # ==== PREPARING DATA === #
    # ======================= #
    log.info('Starting preparing the data.')
        
    # Train data, # change max_length to be computed
    if do_train:
        train_datasets, train_dataloader =  data_loaders.get_prepared_data(input_dir, 'train', mode, batch_size, num_workers = num_workers, nrows_to_load = nrows_to_load)

    # Valid data
    if do_train_with_early_stopping:
        valid_datasets, valid_dataloader =  data_loaders.get_prepared_data(input_dir, 'valid', mode, batch_size, num_workers = num_workers, nrows_to_load = nrows_to_load)

    log.info('Data preparing done.\n')


    # ==== TRAINING ==== #
    # ================== #
    if do_train:
        # Model
        model = encoder_decoder.lstm_seq2seq(device = device, target_len = 2001)
        model.to(device)
    
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
        criterion = nn.MSELoss()
        criterion_valid = nn.MSELoss()
    
        # Store results
        train_results = various_utils.Results()
        valid_results = various_utils.Results()
        
        # Loop over epochs
        for epoch_index in range(0, n_epochs):
            log.info('=========== EPOCH: {0} =========== '.format(epoch_index))
            #if epoch_index%100==0:
            #    log.info('=========== EPOCH: {0} =========== '.format(epoch_index))
    
            # Train
            log.info('=== Training..')
            train_batch_results = various_utils.BatchResults()
            model.train()
            for batch_index, (features, targets) in enumerate(train_dataloader):
                log.debug('Batch_index: {0}'.format(batch_index))
    
                # Put into the correct dimensions for LSTM
                features = features.permute(1,0)
                features = features.unsqueeze(2).to(device)
    
                targets = targets.permute(1,0)
                targets = targets.unsqueeze(2).to(device)
    
                # Get prediction
                out = model(features, targets)
                #log.debug(out.shape)
                
                # Compute loss
                train_loss = criterion(out, targets)
                train_batch_results.loss_total += train_loss.item()
        
                # Backward propagation
                model.zero_grad()
                train_loss.backward()
                optimizer.step()
        
                # Update n_batches
                train_batch_results.n_batches += 1
                
            # Save train results per this epoch
            train_results.store_results_per_epoch(train_batch_results)
                      
            # Validate
            if do_train_with_early_stopping:
                log.info('=== Validating..')
                valid_batch_results = various_utils.BatchResults()
                model.eval()
                with torch.no_grad():
                    for batch_index, (features, targets) in enumerate(train_dataloader):
                        log.debug('Batch_index: {0}'.format(batch_index))
            
                        # Put into the correct dimensions for LSTM
                        features = features.permute(1,0)
                        features = features.unsqueeze(2).to(device)
            
                        targets = targets.permute(1,0)
                        targets = targets.unsqueeze(2).to(device)
                        
                        # Get prediction
                        out = model(features, targets)
                        
                        # Compute loss
                        valid_loss = criterion(out, targets)
                        valid_batch_results.loss_total += valid_loss.item()
                        
                        # Update n_batches
                        valid_batch_results.n_batches += 1
        
                # Save valid results per this epoch
                valid_results.store_results_per_epoch(valid_batch_results)
                    
            # Update LR
            lr = scheduler.get_lr()[0]
            if lr>0.00001:
                    scheduler.step()
                    
            log.info('Epoch: {0}/{1}, Train Loss: {2:.5f},  Valid Loss: {2:.5f}'.format(epoch_index, n_epochs, train_results.loss_history[-1], valid_results.loss_history[-1]))
 
    
# => Compute F1 to decide on the best model  
               
# => TODO: select rows based on speeds
# => TODO: write a function that finds the maximung length for all datasets, here just pad
                      
# => TODO: define a plotter class with save option which can plot stuff in functions
# => TODO: export trained model to onnx
# => TODO: define predict to load the trained model and predict on test data
    # prepare predict method to scale the data using the train scaler