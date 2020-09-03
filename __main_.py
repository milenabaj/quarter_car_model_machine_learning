"""
Main script.

@author: Milena Bajic (DTU Compute)
"""

import sys,os, glob, time
import subprocess
from multiprocessing import cpu_count
import logging
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from utils import data_loaders
from machine_learning_modules import encoder_decoder
from utils import various_utils

if __name__ == "__main__":

    # === SETTINGS === #
    # ================ #

    git_repo_path = subprocess.check_output('git rev-parse --show-toplevel', shell=True, encoding = 'utf-8').strip()
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
    batch_size = 100000 #'full_dataset'
    num_workers = 8 #0
    n_epochs = 1

    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Logger
    log = various_utils.get_main_logger('Main', log_filename = 'info.log', log_file_dir = out_dir)
    log.info('Input dir is: {0}'.format(input_dir))
    log.info('Output dir is: {0}/{1}\n'.format(os.getcwd(), out_dir))

    encoder_decoder.print()


    # ==== TRAINING ====#
    # ===============#
    mode = 'acc-severity'
    train_datasets = data_loaders.get_datasets(input_dir, 'train', mode, batch_size = batch_size, num_workers = num_workers)
    sys.exit(0)
    train_dataset = ConcatDataset(train_datasets)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers=num_workers)

    #valid_datasets = data_loaders.get_datasets(input_dir, 'valid', mode, batch_size = batch_size, num_workers = num_workers)

    # Model
    '''
    model = lstm_seq2seq(device = device, target_len = window_size, with_attention = with_attention)
    model.to(device)

    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    criterion = nn.MSELoss()
    criterion_valid = nn.MSELoss()
    '''
    # Loop over epochs
    for epoch_index in range(0, n_epochs):
        log.info('=========== EPOCH: {0} =========== '.format(epoch_index))
        #if epoch_index%100==0:
        #    log.info('=========== EPOCH: {0} =========== '.format(epoch_index))

        # Loop over batches
        #model.train()
        for batch_index, (features, targets) in enumerate(train_dataloader):
            print('batch_index',batch_index)

            # Put into the correct dimensions for LSTM
            features = features.permute(1,0)
            features = features.unsqueeze(2).to(device)
            #print(features.shape)

            targets = targets.permute(1,0)
            targets = targets.unsqueeze(2).to(device)
            #print(targets.shape)

 # loop over epochs
      # loop over dataset names
          # make a dataset and dataloader with batch_size=full file for each file
            # per each batch do:
            # Backward propagation and weight update
            #model.zero_grad()
            #train_loss.backward()
            #optimizer.step()

