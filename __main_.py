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
from torch.utils.data import Dataset, DataLoader
from utils import data_loaders


if __name__ == "__main__":

    #=== GET ARGUMENTS FROM THE USER ===#
    #===================================#
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
    batch_size = 'full_dataset'
    num_workers = 0
    n_epochs = 1


    #=== LOGGER ===#
    #==============#
    #Create logger
    log = logging.getLogger('Main')
    log.setLevel(logging.DEBUG)
    for h in list(log.handlers):
        log.removeHandler(h)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # Create file handler
    log_filename = '{0}/info.log'.format(out_dir)
    fh = logging.FileHandler(log_filename, mode='w')
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # Make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    log.info('Input dir is: {0}'.format(input_dir))
    log.info('Output dir is: {0}/{1}\n'.format(os.getcwd(), out_dir))


    # ==== TRAIN ====#
    # ===============#
    mode = 'acc-severity'
    train_data = data_loaders.get_datasets(input_dir, 'train', mode, batch_size = batch_size, num_workers = num_workers,
                                           file_handler = fh, formatter = formatter)

    valid_data = data_loaders.get_datasets(input_dir, 'valid', mode, batch_size = batch_size, num_workers = num_workers,
                                           file_handler = fh, formatter = formatter)


    d = train_data[0][2]
    dataset = train_data[0][1]
    for batch_index, (features, targets) in enumerate(d):
       log.info('Batch index')


    sys.exit(0)
    for epoch_index in range(0, n_epochs):
        log.info('=========== EPOCH: {0} =========== '.format(epoch_index))
        #if epoch_index%100==0:
        #    log.info('=========== EPOCH: {0} =========== '.format(epoch_index))

        # Loop over files
        for (filename, dataset, dataloader) in train_data:
            log.info('Filename: {0} '.format(filename))
            for batch_index, (features, targets) in enumerate(dataloader):
                log.info('Batch index: {0}, features: {1}, targets: {2} '.format(batch_index), features[0], targets[0])

    #f = train_dataset.load_file(0)
    #train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers=0)

    #for data in train_dataloader:
        #Data is a list containing 64 (=batch_size) consecutive lines of the file
    #    print(len(data)) #[64,]

    # loop over epochs
      # loop over dataset names
          # make a dataset and dataloader with batch_size=full file for each file
            # per each batch do:
            # Backward propagation and weight update
            #model.zero_grad()
            #train_loss.backward()
            #optimizer.step()

