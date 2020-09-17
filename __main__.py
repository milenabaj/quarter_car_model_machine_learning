"""
Main script.

@author: Milena Bajic (DTU Compute)
"""

import sys,os, glob, time
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
from machine_learning_modules import encoder_decoder, encoder_decoder_with_speed
from utils import various_utils, plot_utils, model_helpers

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(os.getcwd()) 
sys.path.append(os.getenv("HOME"))
sys.path.append('/home/mibaj/') 
   
if __name__ == "__main__":

    # === SETTINGS === #
    # ================ #
    git_repo_path = subprocess.check_output('git rev-parse --show-toplevel', shell=True, encoding = 'utf-8').strip() 
    
    # Script arguments
    parser = argparse.ArgumentParser(description='Please provide command line arguments.')
    
    # Data preparation
    parser.add_argument('--max_length', default = None,
                        help = 'Max length of sequences in train datasets. If None, it will be computed from the datasets. This variable is used for padding.')  
    parser.add_argument('--speed_selection_range', default = [40,41], 
                        help = 'Select datasets for this speed only. Pass None for no selection.') 
    parser.add_argument('--nrows_to_load', default = 100,
                        help = 'Nrows to load from input (use for testing purposes).')
    
    
    # Training and prediction
    parser.add_argument('--do_train', default = True,
                        help = 'Train using the train dataset.')
    parser.add_argument('--model_type', default = 'lstm_encdec',
                        help = 'Choose between lstm_encdec(acceleration sequence -> severity sequence) and lstm_encdec_with_speed(acceleration sequence + speed -> severity sequence).')
    parser.add_argument('--do_train_with_early_stopping', default = True,
                        help = 'Do early stopping using the valid dataset (train flag will be set to true by default).')
    parser.add_argument('--do_test', default = False,
                        help = 'Test on test dataset.')


    # Directories
    parser.add_argument('--input_dir', default = '{0}/data/Golden-car-simulation-August-2020/train-val-test-normalized-split-into-windows-cluster'.format(git_repo_path),
                        help = 'Input directory containing train/valid/test subdirectories with prepared data split into windows.')
    parser.add_argument('--out_dir', default = 'output',
                        help='Output directory for trained models and results.')
    
    # Run on cluster
    parser.add_argument('--run_on_cluster', default = True,
                        help='Overwrite settings for running on cluster.')
 
    # Parse arguments
    args = parser.parse_args()
    if args.model_type not in ['lstm_encdec', 'lstm_encdec_with_speed']:
        sys.exit('Unknown model passed. Choose beteen lstm_encdec and lstm_encdec_with_speed.')  
    model_type = args.model_type
    max_length = args.max_length
    speed_selection_range = args.speed_selection_range  # Use only data with speed in the selected range
    do_train = args.do_train
    do_train_with_early_stopping = args.do_train_with_early_stopping
    do_test = args.do_test
    nrows_to_load = args.nrows_to_load
    input_dir = args.input_dir
    out_dir = args.out_dir
    run_on_cluster = args.run_on_cluster #
    
    # Other settings
    window_size = 2 #   IMPORTANT for plotting!
    acc_to_severity_seq2seq = True # pass True for ac->severity seq2seq or False to do acc->class 
    batch_size = 24
    num_workers = 0 #0
    n_epochs = 1
    learning_rate= 0.001
    patience = 30
    n_pred_plots = 5
    save_results = True
        
    # ======== SET ========= #
    # ======================= #
    # If run on cluster
    if run_on_cluster:
        input_dir = '/dtu-compute/mibaj/Golden-car-simulation-August-2020/train-val-test-normalized-split-into-windows-size-{0}'.format(window_size)
        out_dir = '/dtu-compute/mibaj/Golden-car-simulation-August-2020' 
        nrows_to_load = -1
        batch_size = 512
        do_test = False
        n_epochs = 200
        n_pred_plots = 100

    model_name = model_helpers.get_model_name(model_type)
    
    # Set flags
    if do_train_with_early_stopping: 
        do_train=True
        
    # Create output directory    
    if speed_selection_range:
        out_dir = '{0}_windowsize_{1}_speedrange_{2}_{3}_{4}_{5}'.format(out_dir, window_size, speed_selection_range[0], speed_selection_range[1], model_name, device)
    else:
        out_dir = '{0}_{1}_{2}'.format(out_dir, model_name, device)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # Logger
    log = various_utils.get_main_logger('Main', log_filename = 'info.log', log_file_dir = out_dir)
    log.info('Output dir is: {0}\n'.format(out_dir))


    # ==== PREPARING DATA === #
    # ======================= #
    log.info('Starting preparing the data.\n')
        
    if not max_length:
         max_length_train = data_loaders.get_dataset_max_length(input_dir, 'train', num_workers = 0,  speed_selection_range =  speed_selection_range, 
                                                          nrows_to_load = nrows_to_load)
         max_length_valid = data_loaders.get_dataset_max_length(input_dir, 'valid', num_workers = 0,  speed_selection_range =  speed_selection_range, 
                                                          nrows_to_load = nrows_to_load)
         max_length = np.max([max_length_train, max_length_valid])
         
    # Train data, # change max_length to be computed
    if do_train:
        train_datasets, train_dataloader =  data_loaders.get_prepared_data(input_dir, 'train', acc_to_severity_seq2seq, batch_size, num_workers = num_workers, 
                                                                           max_length = max_length, speed_selection_range =  speed_selection_range,  
                                                                           nrows_to_load = nrows_to_load)

    # Valid data
    if do_train_with_early_stopping:
        valid_datasets, valid_dataloader =  data_loaders.get_prepared_data(input_dir, 'valid', acc_to_severity_seq2seq, batch_size, num_workers = num_workers, 
                                                                           max_length = max_length,  speed_selection_range =  speed_selection_range,
                                                                           nrows_to_load = nrows_to_load)
        
    # Test data
    if do_test:
        test_datasets, test_dataloader =  data_loaders.get_prepared_data(input_dir, 'test', acc_to_severity_seq2seq, batch_size, num_workers = num_workers, 
                                                                           max_length = max_length,  speed_selection_range =  speed_selection_range,
                                                                           nrows_to_load = nrows_to_load)
    
    log.info('Data preparing done.\n')

    # ==== TRAINING ==== #
    # ================== #
    if do_train:
        
        # Model
        if model_type=='lstm_encdec':
            model = encoder_decoder.lstm_seq2seq(device = device, target_len = max_length, use_teacher_forcing = True)
        elif model_type=='lstm_encdec_with_speed':
            model = encoder_decoder_with_speed.lstm_seq2seq_with_speed(device = device, target_len = max_length, use_teacher_forcing = True)  
        model.to(device)

        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
        criterion = nn.MSELoss()
        criterion_valid = nn.MSELoss()
    
        # Store results
        train_results = model_helpers.Results()
        valid_results = model_helpers.Results()
        
        # Early_stopping 
        early_stopping = model_helpers.EarlyStopping(patience = patience)

        # Loop over epochs
        for epoch_index in range(0, n_epochs):
            log.info('=========== EPOCH: {0} =========== '.format(epoch_index))
            #if epoch_index%100==0:
            #    log.info('=========== EPOCH: {0} =========== '.format(epoch_index))
    
            # Train
            log.info('=== Training..')
            train_batch_results = model_helpers.BatchResults()
            model.train()
            for batch_index, (acc, speed, orig_length, targets) in enumerate(train_dataloader):
                #log.debug('Batch_index: {0}'.format(batch_index))
    
                # Put into the correct dimensions for LSTM
                acc = acc.permute(1,0) 
                acc = acc.unsqueeze(2).to(device)
    
                targets = targets.permute(1,0)
                targets = targets.unsqueeze(2).to(device)
    
                
                # Get prediction
                if model_type=='lstm_encdec':
                    out = model(acc, targets)
                elif model_type=='lstm_encdec_with_speed':
                    speed = speed.reshape(1,acc.shape[1], 1).to(device)              
                    out = model(acc, speed, targets)
    
                #sys.exit(0)
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
                valid_batch_results = model_helpers.BatchResults()
                model.eval()
                with torch.no_grad():
                    for batch_index, (acc, speed, orig_length, targets) in enumerate(train_dataloader):
                        #log.debug('Batch_index: {0}'.format(batch_index))
            
                        # Put into the correct dimensions for LSTM
                        acc = acc.permute(1,0) 
                        acc = acc.unsqueeze(2).to(device)
            
                        targets = targets.permute(1,0)
                        targets = targets.unsqueeze(2).to(device)
            
                        #log.debug('acc shape: {0}, speed shape: {1}'.format(acc.shape, speed.shape))
                        
                        # Get prediction
                        if model_type=='lstm_encdec':
                            out = model(acc, targets)
                        elif model_type=='lstm_encdec_with_speed':
                            speed = speed.reshape(1,acc.shape[1], 1).to(device)             
                            out = model(acc, speed, targets)
                        
                        # Compute loss
                        valid_loss = criterion(out, targets)
                        valid_batch_results.loss_total += valid_loss.item()
                        
                        # Update n_batches
                        valid_batch_results.n_batches += 1
        
                # Save valid results per this epoch
                valid_results.store_results_per_epoch(valid_batch_results)
                
                # Check for early stopping
                early_stopping.check_this_epoch(valid_loss = valid_results.loss_history[-1], train_loss = train_results.loss_history[-1], 
                                                curr_epoch = epoch_index, state_dict = model.state_dict())  
        
                if early_stopping.early_stop:
                    log.info("Early stopping")
                    log.info('Epoch: {0}/{1}, Train Loss: {2:.6f},  Valid Loss: {2:.6f}'.format(epoch_index, n_epochs, train_results.loss_history[-1], valid_results.loss_history[-1]))
                    break
                
            # Update LR
            lr = scheduler.get_lr()[0]
            if lr>0.00001:
                    scheduler.step()
                    
            log.info('Epoch: {0}/{1}, Train Loss: {2:.5f},  Valid Loss: {2:.5f}'.format(epoch_index, n_epochs, train_results.loss_history[-1], valid_results.loss_history[-1]))


# ======== BEST MODEL PREDICTIONS ========= #
# ========================================= #
# Onnx input
if model_type=='lstm_encdec':
    onnx_input = (acc)
elif model_type=='lstm_encdec_with_speed':            
    onnx_input = (acc, speed) #saved is without teacher forcing, output is not needed for prediction only the shape is needed for model structure
    
# Best Model (saved as .pth and .onnx)
best_model_info = model_helpers.ModelInfo(model, early_stopping = early_stopping, model_type = model_type, onnx_input = onnx_input, out_dir = out_dir)
log.debug('Best epoch: {0}\n'.format(best_model_info.epoch))

# Best Model Predictions
if do_train:
    train_true, train_pred, train_speeds, train_orig_lengths, train_loss = best_model_info.predict(train_dataloader, datatype = 'train')
if do_train_with_early_stopping:
    valid_true, valid_pred, valid_speeds, valid_orig_lengths, valid_loss = best_model_info.predict(valid_dataloader, datatype = 'valid')
if do_test:
    test_true, test_pred, test_speeds, test_orig_lengths, test_loss = best_model_info.predict(test_dataloader, datatype = 'test')

# Plot results 
if (do_train_with_early_stopping and do_test):
    plotter = plot_utils.Plotter(train_results = train_results, valid_results = valid_results, window_size = window_size, speed_selection = speed_selection_range, save_plots = save_results, model_name = model_name, out_dir = out_dir)
    plotter.plot_trainvalid_learning_curve()
    plotter.plot_pred_vs_true_timeseries(train_true, train_pred, train_speeds, train_orig_lengths, 'train', n_examples = n_pred_plots)
    plotter.plot_pred_vs_true_timeseries(valid_true, valid_pred, valid_speeds, valid_orig_lengths, ' valid', n_examples = n_pred_plots)
    plotter.plot_pred_vs_true_timeseries(test_true, test_pred, test_speeds, test_orig_lengths, 'test', n_examples = n_pred_plots)
    
elif (do_train_with_early_stopping and not do_test):
    plotter = plot_utils.Plotter(train_results = train_results, valid_results = valid_results, window_size = window_size, speed_selection = speed_selection_range, 
                                 save_plots = save_results, model_name = model_name, out_dir = out_dir)
    plotter.plot_trainvalid_learning_curve()
    plotter.plot_pred_vs_true_timeseries(train_true, train_pred, train_speeds, train_orig_lengths, 'train', n_examples= n_pred_plots)
    plotter.plot_pred_vs_true_timeseries(valid_true, valid_pred, valid_speeds, valid_orig_lengths, ' valid', n_examples= n_pred_plots)
    
elif (not do_train_with_early_stopping and do_test):
    plotter = plot_utils.Plotter(window_size = window_size, speed_selection = speed_selection_range, save_plots = save_results, model_name = model_name, out_dir = out_dir)
    plotter.plot_trainvalid_learning_curve()
    plotter.plot_pred_vs_true_timeseries(test_true, test_pred, test_speeds, test_orig_lengths, 'test', n_examples= n_pred_plots)


log.info('Results saved to: {0}'.format(out_dir))

# => TODO: define predict to load the trained model and predict on test data
    # prepare predict method to scale the data using the train scaler

