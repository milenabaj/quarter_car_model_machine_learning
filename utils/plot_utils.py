"""
Plot utils.

@author: Milena Bajic (DTU Compute)
"""
import os, glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from quarter_car_model_machine_learning.utils.various_utils import *
import torch


# Get logger for module
plog = get_mogule_logger("plot_utils")
plt.rc('font', size=30)

def plot_learning_curve(x, y, y_label = 'Loss', model_name = '', plotname_suffix = '',text = '', out_dir = '.', is_cuda=False):
    if is_cuda:
        plt.figure(figsize=(15,15))
        plt.rc('font', size=LARGE_SIZE)
    else:
        plt.figure(figsize=(5,5))
        plt.rc('font', size=SMALL_SIZE)
    plt.plot(x, label='Train Loss', color = 'b',  marker='.', markersize=4, linewidth = 0.9)
    plt.plot(y, label='Valid Loss', color = 'r',  marker='.', markersize=4, linewidth = 0.9)
    plt.title(model_name)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    #plt.xlim([0., plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]*1.2])
    plt.text(0.1, 0.1, text, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.legend()
    plot_name = get_plotname(y_label, plotname_suffix)
    #plt.gca().xaxis.set_major_locator((MaxNLocator(integer=True)))
    #print(plot_name)a
    plt.savefig('{0}/{1}.png'.format(out_dir, plot_name))
    plt.show()
    return


class Plotter():
    def __init__(self, train_results = None, valid_results = None, test_results = None, window_size = None,
                 model_name = '', attn_type = None, speed_selection = None, save_plots = False,  out_dir = '.'):
        plog.info('===>\nCreating plotter')
        self.train_results = train_results 
        self.valid_results = valid_results
        self.window_size = window_size
        self.test_results = test_results
        self.save_plots = save_plots
        self.model_name = model_name
        self.attn_type = attn_type
        self.speed_selection = speed_selection
        
        # Output directory
        self.out_dir = '{0}/plots'.format(out_dir)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        
    def plot_trainvalid_learning_curve(self, plot_text = ''):   
        plog.debug('Plotting Learning Curve')
        train_loss = self.train_results.loss_history
        valid_loss = self.valid_results.loss_history
        plt.figure(figsize=(50,50))
        plt.rc('font', size=30)
        plt.plot(train_loss, label='Train Loss', color = 'b',  marker='.', markersize=35, linewidth = 0.9)
        plt.plot(valid_loss, label='Valid Loss', color = 'r',  marker='.', markersize=35, linewidth = 0.9)
        plt.title('Learning Curve: {0}'.format(self.model_name))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.xlim([0., plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]*1.2])
        plt.text(0.1, 0.7, plot_text, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        plt.legend()
        plot_name = self.model_name + '_learning_curve'
        plt.savefig('{0}/{1}.png'.format(self.out_dir, plot_name))
        plt.savefig('{0}/{1}.pdf'.format(self.out_dir, plot_name))
        plt.show()
        return

    def plot_pred_vs_true_timeseries(self, true_batch, pred_batch, attentions, speeds, orig_lengths, dataset_type, batch_index_to_plot = 0,  n_examples = 4):
        import random
        plog.debug('Plotting predicted vs true timeseries plot for {0} dataset.'.format(dataset_type))
        plt.rc('font', size=20)
        
        # Plot batch = batch_index_to_plot
        true_batch= true_batch[batch_index_to_plot]
        pred_batch = pred_batch[batch_index_to_plot]
        speeds = speeds[batch_index_to_plot]
        orig_lengths = orig_lengths[batch_index_to_plot]
        attention = attentions[batch_index_to_plot]
        
        # Samples to plot
        n_samples = true_batch.shape[1] # batch size
        if n_examples>n_samples: #asked to plot more examples than there are samples in the batch
            n_examples=n_samples
        random.seed(123)
        examples = random.sample(range(n_samples), n_examples) #sample n_examples number of sequences from the batch
      
        for i, example in enumerate(examples):
            
            # Unpad
            orig_length = orig_lengths[example]
            speed = speeds[example]
            pred = 100*pred_batch[:,example,:].reshape(-1)[:orig_length] # to cm
            true = 100*true_batch[:,example,:].reshape(-1)[:orig_length] # to cm
            distance = np.linspace(0,self.window_size, num = orig_length)
            attn = attention[:,:,example] #first index is output ts, second is input ts
                   
            # Plot
            fig, ax = plt.subplots()
            ax.scatter(distance, pred, c = 'blue', label = 'Predicted', s=8, marker='*', alpha=0.9)
            ax.scatter(distance, true, c = 'red', label = 'True', s=8, marker='o', alpha=0.9)
            #fig.set_title(dataset_type)
            ax.set_ylabel('Severity [cm]')
            ax.set_xlabel('Distance [m]')
            #ax.set_ylim(( ax.get_ylim()[0]-1, ax.get_ylim()[1]+1 ))
            #ax.yaxis.set_major_locator(MultipleLocator(10))
            #ax.yaxis.set_minor_locator(MultipleLocator(2))
            ax.grid()
            
            # Text
            unit= r'$\frac{km}{h}$'
            dataset_type = dataset_type.capitalize()
            text = '{0} dataset\nSpeed = {1}{2}'.format(dataset_type, speed, unit)
            #ax.text(0.7,0.15, text, fontsize=35, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
            #ax.scatter([], [], ' ', label='{0} dataset, speed = {')
            
            #ax.legend(fontsize=45,  loc='lower left', prop={'size': 50})
            #ax.legend()
            plt.tight_layout()
            
            # Text
            #ax.text(0.8, 0.8, '-{0} dataset \nSpeed: {1} km/h'.format(dataset_type.upper(), speed), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
            #fig.tight_layout()
  
            # Save
            if self.speed_selection:
                figname = '{0}/{1}_{2}_{3}_speedsel_{4}_{5}_severity_figure_{6}.png'.format(self.out_dir, dataset_type.lower(), self.model_name, self.attn_type,
                                                                                       self.speed_selection[0], self.speed_selection[1], example)
            else:
                figname = '{0}/{1}_{2}_{3}_fullspeed_severity_figure{4}.png'.format(self.out_dir, dataset_type, self.model_name, self.attn_type, example)
                
            plt.savefig(figname)
            plt.savefig(figname.replace('.png','.pdf'))
            
            plt.show()
            plt.tight_layout()
            plt.close('all')
            
                        
            # Plot Attention
            self.plot_attention(attn, attention_type=self.attn_type, figname = figname.replace('severity','attention'))
          
            #sys.exit(0)
            
        return
    
    def plot_attention(self, attentions, attention_type, figname):
        plt.rc('font', size=20)
        fig, ax = plt.subplots()
        #plt.title('Attention weights')
        plt.minorticks_on()
        
        # We change the fontsize of minor ticks label 
        #ax.tick_params(axis='both', which='major', labelsize=10)
        #ax.tick_params(axis='both', which='minor', labelsize=5)
        
        c = ax.imshow(attentions)
        plt.colorbar(c)
        
        # Save fig        
        plt.savefig(figname)
        plt.savefig(figname.replace('.png','.pdf'))
        plt.show()
        plt.close('all')           
        
        return