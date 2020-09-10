#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:41:17 2020

@author: mibaj
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from quarter_car_model_machine_learning.utils.various_utils import *
from quarter_car_model_machine_learning.machine_learning_modules import encoder_decoder
import torch

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get logger for module
hlog = get_mogule_logger("model_helpers")


class Results:
    def __init__(self):
        self.loss_history = []
        
    def store_results_per_epoch(self, batch_results):
        loss_epoch = float(batch_results.loss_total/batch_results.n_batches)
        self.loss_history.append(loss_epoch)
        
class BatchResults:
    def __init__(self):   
        self.loss_total = 0
        self.n_batches = 0 
        
        

class EarlyStopping:
    def __init__(self, patience):
        self.min_valid_loss = np.Inf
        self.patience = patience
        self.early_stop = False
                
        self.best_epoch = np.Inf
        self.best_valid_loss = np.Inf
        self.best_train_loss = np.Inf
        self.best_state_dict  = np.Inf 
        
    def check_this_epoch(self, valid_loss, train_loss, curr_epoch, state_dict):
        is_better = False
        
        # Check is this epoch is better
        if valid_loss< self.min_valid_loss:
            self.min_valid_loss = valid_loss
            is_better = True
            self.best_epoch = curr_epoch
            self.best_valid_loss = valid_loss
            self.best_train_loss = train_loss
            self.best_state_dict = deepcopy(state_dict)
            
        if (curr_epoch-self.best_epoch)==self.patience:
            self.early_stop = True
            
        log_vu.debug('Checking epoch {0} .. better than before: {1}'.format(curr_epoch, is_better))
        return is_better
    
 
class ModelInfo:
    def __init__(self, max_length, state_dict, early_stopping = None, print_model_info = True, model_name = ''):
        
        # Best Model
        self.model = encoder_decoder.lstm_seq2seq(device = device, target_len = max_length)
        self.model.to(device)
        self.model.load_state_dict(state_dict)
        
        # Other info 
        self.model_name = model_name
        if early_stopping:
            self.epoch = early_stopping.best_epoch
            self.train_loss = early_stopping.best_train_loss 
            self.valid_loss = early_stopping.best_valid_loss
            
        if print_model_info:
            self.print_model_info
            
    def print_model_info(self):
        from pprint import pprint
        to_print = {k:v for k,v in vars(best_model_info).items() if k!='best_model_info'}
        log_vu('Model {0}: {1}'.format(self.model_name, pprint(to_print)))

    def predict(self, dataloader, n_batches = 1, datatype = ''):
        '''
        Get model predictions on n_batches of the dataloader.
        '''
        from statistics import mean
        log_vu.info('\n===> Predicting {0}'.format(datatype))
        losses = []
        predictions = [] 
        
        criterion = nn.MSELoss()
        self.model.eval()
        with torch.no_grad():
            for batch_index, (features, targets) in enumerate(dataloader):
                log_vu.debug('Batch_index: {0}'.format(batch_index))
    
                # Put into the correct dimensions for LSTM
                features = features.permute(1,0)
                features = features.unsqueeze(2).to(device)
    
                targets = targets.permute(1,0)
                targets = targets.unsqueeze(2).to(device)
                
                # Get prediction
                out = self.model(features, targets)
                predictions.append(out)
                
                # Compute loss
                loss = criterion(out, targets)
                losses.append(loss.item())
                
                # Update n_batches
                if batch_index == n_batches-1: #batch_index starts at 0
                    break
                
        mean_loss = mean(losses)
        return predictions, mean_loss