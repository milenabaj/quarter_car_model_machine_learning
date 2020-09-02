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
    batch_size = 1


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


    # ==== MAIN  ====#
    # ===============#
    train_dataset = data_loaders.Dataset(input_dir_base = input_dir, filetype = 'train', batch_size = batch_size,
                                         file_handler = fh, formatter = formatter)
    #f = train_dataset.load_file(0)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers=cpu_count())

    # loop over epochs
      # loop over dataset names
          # make a dataset and dataloader with batch_size=full file for each file
            # per each batch do:
            # Backward propagation and weight update
            #model.zero_grad()
            #train_loss.backward()
            #optimizer.step()

