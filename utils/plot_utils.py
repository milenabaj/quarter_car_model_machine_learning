"""
Plot utils.

@author: Milena Bajic (DTU Compute)
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from quarter_car_model_machine_learning.utils.various_utils import *
import torch

# Get logger for module
plog = get_mogule_logger("plot_utils")


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
    def __init__(self, train_results = None, valid_results = None, test_results = None, model_name = '',
                 save_plots = False,  out_dir = '.'):
        plog.info('===>\nCreating plotter')
        self.train_results = train_results 
        self.valid_results = valid_results
        self.test_results = test_results
        self.save_plots = save_plots
        self.model_name = model_name
        
        # Output directory
        self.out_dir = '{0}/plots'.format(out_dir)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        
    def plot_trainvalid_learning_curve(self, plot_text = ''):   
        plt.figure(figsize=(20,20))
        plt.rc('font', size=30)
        plt.plot(self.train_results.loss_history, label='Train Loss', color = 'b',  marker='.', markersize=16, linewidth = 0.9)
        plt.plot(self.valid_results.loss_history, label='Valid Loss', color = 'r',  marker='.', markersize=16, linewidth = 0.9)
        plt.title('Learning Curve: {0}'.format(self.model_name))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.xlim([0., plt.xlim()[1]])
        plt.ylim([0, plt.ylim()[1]*1.2])
        plt.text(0.1, 0.1, plot_text, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        plt.legend()
        plot_name = self.model_name + '_learning_curve'
        plt.savefig('{0}/{1}.png'.format(self.out_dir, plot_name))
        plt.savefig('{0}/{1}.pdf'.format(self.out_dir, plot_name))
        plt.show()
        return
    '''
    def plot_predicted_vs_true_severity(self, plot_text = ''):
        
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
                '''
    def plot_all(self):
        self.plot_trainvalid_learning_curve()