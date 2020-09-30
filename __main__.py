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
        nrows_to_load = 1000
        defect_height_selection = [-10,10]
        defect_width_selection = [0,20]
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
                                                                           max_length = max_length, speed_selection_range =  speed_selection_range,  
                                                                           nrows_to_load = nrows_to_load, defect_height_selection = defect_height_selection, defect_width_selection = defect_width_selection)
        

    # Valid data
    if do_train_with_early_stopping:
        valid_dataset, valid_dataloader, _ =  data_loaders.get_prepared_data(input_dir, out_dir, 'valid', acc_to_severity_seq2seq, batch_size, num_workers = num_workers, 
                                                                           max_length = max_length,  speed_selection_range =  speed_selection_range,
                                                                           nrows_to_load = nrows_to_load, defect_height_selection = defect_height_selection, defect_width_selection = defect_width_selection)
        
    # Test data
    if do_test:
        test_dataset, test_dataloader, _ =  data_loaders.get_prepared_data(input_dir, out_dir, 'test', acc_to_severity_seq2seq, batch_size, num_workers = num_workers, 
                                                                           max_length = max_length,  speed_selection_range =  speed_selection_range,
                                                                           nrows_to_load = nrows_to_load, defect_height_selection = defect_height_selection, defect_width_selection = defect_width_selection)
    
    log.info('Data preparing done.\n')
    
    # ==== TRAINING ==== #
    # ================== #
    if do_train:
        
        # Model
        if model_type=='lstm_encdec':
            model = encoder_decoder.lstm_seq2seq(device = device, target_len = max_length, use_teacher_forcing = True)
        elif model_type=='lstm_encdec_with_attn':
            model = encoder_decoder_with_attention.lstm_seq2seq_with_attn(device = device, target_len = max_length, use_teacher_forcing = True)
        elif model_type=='lstm_encdec_with_speed':
            model = encoder_decoder_with_speed.lstm_seq2seq_with_speed(device = device, target_len = max_length, use_teacher_forcing = True)  
        model.to(device)

        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
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
            for batch_index, (acc, scaled_speed, speed, orig_length, targets) in enumerate(train_dataloader):
                #log.debug('Batch_index: {0}'.format(batch_index))
    
                # Put into the correct dimensions for LSTM
                acc = acc.permute(1,0) 
                acc = acc.unsqueeze(2).to(device)
        
                targets = targets.permute(1,0)
                targets = targets.unsqueeze(2).to(device)
                
                if model_type=='lstm_encdec' :
                    out = model(acc, targets)
                    
                elif model_type=='lstm_encdec_with_attn':       
                    out = model(acc, targets)
                    
                elif model_type=='lstm_encdec_with_speed':
                    scaled_speed = scaled_speed.reshape(acc.shape[1],1).to(device)
                    out = model(acc, scaled_speed, targets)
                    
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
                
                #d = model.decoder
                #e = model.encoder
                #sys.exit(0)
            # Save train results per this epoch
            train_results.store_results_per_epoch(train_batch_results)
                      
            # Validate
            if do_train_with_early_stopping:
                log.info('=== Validating..')
                valid_batch_results = model_helpers.BatchResults()
                model.eval()
                with torch.no_grad():
                    for batch_index, (acc, scaled_speed, speed, orig_length, targets) in enumerate(train_dataloader):
                        #log.debug('Batch_index: {0}'.format(batch_index))
                        # Put into the correct dimensions for LSTM
                        acc = acc.permute(1,0) 
                        acc = acc.unsqueeze(2).to(device)
                
                        targets = targets.permute(1,0)
                        targets = targets.unsqueeze(2).to(device)
                        
                        if model_type=='lstm_encdec' :
                            out = model(acc, targets)
                            
                        elif model_type=='lstm_encdec_with_attn':       
                            out = model(acc, targets)
                            
                        elif model_type=='lstm_encdec_with_speed':
                            scaled_speed = scaled_speed.reshape(acc.shape[1],1).to(device)
                            out = model(acc, scaled_speed, targets)
                                   
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
                    
            log.info('Epoch: {0}/{1}, Train Loss: {2:.7f},  Valid Loss: {2:.7f}'.format(epoch_index, n_epochs, train_results.loss_history[-1], valid_results.loss_history[-1]))


# ======== BEST MODEL PREDICTIONS ========= #
# ========================================= #
# Onnx input
if model_type=='lstm_encdec' or model_type=='lstm_encdec_with_attn':
    onnx_input = (acc)
elif model_type=='lstm_encdec_with_speed':            
    onnx_input = (acc, scaled_speed) #saved is without teacher forcing, output is not needed for prediction only the shape is needed for model structure
    
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

