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
        if valid_loss < self.min_valid_loss:
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
    def __init__(self, model, early_stopping = None, print_model_info = True,  onnx_input = None, out_dir = '.', save = True, model_type = ''):
        
        # Best Model
        self.model = model
        self.model.use_teacher_forcing = False #turn of for predictions
        self.model.to(device)
        self.model_type = model_type
        self.model_name = get_model_name(self.model_type)
        self.out_dir = out_dir
        self.onnx_input = onnx_input
        
        if early_stopping:
            self.epoch = early_stopping.best_epoch
            self.train_loss = early_stopping.best_train_loss 
            self.valid_loss = early_stopping.best_valid_loss
            
        if print_model_info:
            self.print_model_info
            
        if save:
            self.save_model()
        
        
    def print_model_info(self):
        from pprint import pprint
        to_print = {k:v for k,v in vars(best_model_info).items() if k!='best_model_info'}
        log_vu('Model {0}: {1}'.format(self.model_name, pprint(to_print)))
    
    def save_model(self):    
        path = '{0}/trained_model_{1}.pt'.format(self.out_dir, self.model_name)
        torch.save(self.model, path)
        log_vu.info('Saved model as: {0}'.format(path))
    
        # Onnx
        onnx_path = path.replace('.pt','.onnx')
        torch.onnx.export(self.model, self.onnx_input, onnx_path, opset_version = 11)
        log_vu.info('Saved model as: {0}'.format(onnx_path))

    def predict(self, dataloader, n_batches = -1, datatype = ''):
        '''
        Get model predictions on n_batches of the dataloader.
        '''
        from statistics import mean
        log_vu.info('\n===> Predicting {0}'.format(datatype))
        
        losses = []
        predicted_targets = [] 
        true_targets = []
        speeds = []
        orig_lengths = []
        attentions = {}
        
        criterion = nn.MSELoss()
        self.model.eval()
        with torch.no_grad():
            for batch_index, (acc, scaled_speed, speed,  orig_length, targets) in enumerate(dataloader):
                log_vu.debug('Batch_index: {0}'.format(batch_index))
                
                # Put into the correct dimensions for LSTM
                acc = acc.permute(1,0) 
                acc = acc.unsqueeze(2).to(device)
        
                targets = targets.permute(1,0)
                targets = targets.unsqueeze(2).to(device)
                
                if self.model_type=='lstm_encdec' :
                    out = self.model(acc, targets)
                    
                elif self.model_type=='lstm_encdec_with_attn':       
                    out = self.model(acc, targets)
                    attentions[batch_index] = self.model.attention_weights.cpu().detach().numpy()
                    
                elif self.model_type=='lstm_encdec_with_speed':
                    scaled_speed = scaled_speed.reshape(acc.shape[1],1).to(device)
                    out = self.model(acc, scaled_speed, targets)
                                         
                predicted_targets.append(out.cpu().detach().numpy())                    
                true_targets.append(targets.cpu().detach().numpy())
                
                # Compute loss
                loss = criterion(out, targets)
                losses.append(loss.item())
                
                # Save speeds and original lengths
                speeds.append(speed.cpu().detach().numpy())
                orig_lengths.append(orig_length.cpu().detach().numpy())
                
                # Plot features
                if datatype=='train' and batch_index==0:
                    plt.rc('font', size=12)
                    n_examples= 3
                    n_features = self.model.decoder.hidden_size
                    for example in range(n_examples):
                        fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
                        fig.suptitle(str(datatype) + ' data, example: ' + str(example), fontsize=12)
                        ax1.plot(acc[:,example,0].cpu().detach().numpy())
                        ax1.plot(targets[:,example,0].cpu().detach().numpy())
                        for feature_idx in range(n_features):
                            feat = self.model.encoder.lstm_out.cpu().detach().numpy()[:,example,feature_idx]
                            ax2.plot(feat, linewidth=0.5)
                        plt.tight_layout()
                        figname = '{0}/{1}_{2}_features_example{3}.png'.format(self.out_dir, datatype, self.model_name, example)
                        print('Saving: ',figname)
                        plt.savefig(figname)
                        plt.show()
                #log.debug(out.shape)
                #sys.exit(0)
                plt.close('all')
                
                # Update n_batches
                if batch_index == n_batches-1: # batch_index starts at 0
                    break
                
        mean_loss = mean(losses)
        return true_targets, predicted_targets, attentions, speeds, orig_lengths, mean_loss

def get_model_name(model_type):
    if model_type=='lstm_encdec':
        return 'LSTM_encoder_decoder_acc'
    elif model_type=='lstm_encdec_with_attn':
        return 'LSTM_encoder_decoder_acc_attn'
    elif model_type=='lstm_encdec_with_speed':
        return 'LSTM_encoder_decoder_accspeed'

    