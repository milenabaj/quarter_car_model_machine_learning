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
    parser.add_argument('--load_subset', default = False,
                        help = 'If load_subset is true, will process 100 rows only (use for testing purposes).')
    parser.add_argument('--input_dir', default = '{0}/data/Golden-car-simulation-August-2020/train-val-test-normalized-split-into-windows-cluster'.format(git_repo_path),
                        help = 'Input directory containing train/valid/test subdirectories with prepared data split into windows.')
    parser.add_argument('--output_dir', default = 'output',
                        help='Output directory for trained models and results.')

    # Parse arguments
    args = parser.parse_args()
    load_subset = args.load_subset
    input_dir = args.input_dir
    out_dir = args.output_dir

    # Other settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = 'acc-severity'
    batch_size = 10 #'full_dataset'
    num_workers = 0 #0
    n_epochs = 1
    learning_rate= 0.001

    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Logger
    log = various_utils.get_main_logger('Main', log_filename = 'info.log', log_file_dir = out_dir)
    log.info('Input dir is: {0}'.format(input_dir))
    log.info('Output dir is: {0}/{1}\n'.format(os.getcwd(), out_dir))

    # ==== PREPARING DATA === #
    # ======================= #
    train_datasets = data_loaders.get_datasets(input_dir, 'train', mode, batch_size = batch_size, num_workers = num_workers)
    train_dataset = ConcatDataset(train_datasets)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers)

    #valid_datasets = data_loaders.get_datasets(input_dir, 'valid', mode, batch_size = batch_size, num_workers = num_workers)


    # ==== TRAINING ==== #
    # ================== #
    # Model
    model = encoder_decoder.lstm_seq2seq(device = device, target_len = 2001)
    model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    criterion = nn.MSELoss()
    criterion_valid = nn.MSELoss()

    # Loop over epochs
    train_loss_history = []
    train_loss_min = 100000
    for epoch_index in range(0, n_epochs):
        log.info('=========== EPOCH: {0} =========== '.format(epoch_index))
        
        #if epoch_index%100==0:
        #    log.info('=========== EPOCH: {0} =========== '.format(epoch_index))

        # Loop over batches
        train_loss_total = 0
        n_batches = 0
        
        model.train()
        for batch_index, (features, targets) in enumerate(train_dataloader):
            log.debug('Batch_index: {0}'.format(batch_index))

            # Put into the correct dimensions for LSTM
            features = features.permute(1,0)
            features = features.unsqueeze(2).to(device)

            targets = targets.permute(1,0)
            targets = targets.unsqueeze(2).to(device) #(seq_length, batch_size, input_features)

            # Predict
            out = model(features, targets)
            
            # Compute loss
            train_loss = criterion(out, targets)
            train_loss_total += train_loss.item()
    
            # Backward propagation
            model.zero_grad()
            train_loss.backward()
            optimizer.step()
    
            n_batches += 1
            sys.exit(0)
  
        train_loss_epoch = float(train_loss_total)/n_batches
        train_loss_history.append(train_loss_epoch)
    
        lr = scheduler.get_lr()[0]
        log.info('Epoch: {0}/{1}, Train Loss: {2:.5f}, lr: {3}'.format(epoch_index, n_epochs, train_loss_epoch, lr))
    
        # Update LR
        if lr>0.00001:
            scheduler.step()

    # TODO: Validate model 
            
    sys.exit(0)
    # ==== PLOTTING RESULTS ==== #
    # ========================== #
    log.info('Plotting')
    plt.rc('font', size=20)
    
    # Distance
    dt = round(full_data.times[0,0,2]-full_data.times[0,0,1],3)
    time_start_0 = full_data.times[0] - full_data.times[0,:,0] + dt
    distance = v*time_start_0
    
    
    # Samples to plot
    n_samples = out.cpu().detach().numpy().shape[1]
    random.seed(123)
    examples = random.sample(range(n_samples), n_plots)
    
    for i, example in enumerate(examples):
        pred = 100*out.cpu().detach().numpy()[:,example,:].reshape(-1)
        true = 100*train_targets.cpu().detach().numpy()[:,example,:].reshape(-1)
    
        save_fig = False
        if i==0 or i%4==0:
            fig, axis = plt.subplots(2, 2, figsize = (20,20))
            ax = axis[0,0]
            fig_i = int(i/4)
        else:
            if i == fig_i*4 + 1:
                ax = axis[0,1]
            elif i == fig_i*4 + 2:
                ax = axis[1,0]
            elif i == fig_i*4 + 3:
                ax = axis[1,1]
                save_fig = True
    
    
        ax.scatter(distance, pred, c = 'blue', label = 'Predicted', s=18, marker='*', alpha=0.5)
        ax.scatter(distance, true, c = 'red', label = 'True', s=30, marker='o', alpha=0.5)
        #ax.set_title('Severity')
        ax.set_ylabel('Severity [cm]')
        ax.set_xlabel('Distance [m]')
        #ax.set_ylim(( ax.get_ylim()[0]-1, ax.get_ylim()[1]+1 ))
        ax.legend(fontsize=20,  prop={'size': 20})
        ax.grid()
        #ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(2))
    
        if save_fig:
            plt.savefig('{0}/{1}_figure{2}_{3}.png'.format(out_dir, 'severity', fig_i, plot_string))
            plt.close('all')
    
    fig = plt.figure(figsize = (20,20))
    ax = fig.add_subplot(111)
    train_loss_history = 1000*train_loss_history
    plt.plot(train_loss_history, label = 'Train Loss',  color = 'b',  marker='.', markersize=6, linewidth = 0.9)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.grid()
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.savefig('{0}/{1}_{2}.png'.format(out_dir, 'lstm_train_loss',  plot_string))
    
    
    log.info('Number of train samples: {0}'.format(train_loader.dataset.n_samples))
    log.info('Output directory: {0}'.format(out_dir))
    
