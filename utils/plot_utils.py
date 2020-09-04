"""
PyTorch Data loading utils.

@author: Milena Bajic (DTU Compute)
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Get logger for module
dlog = get_mogule_logger("plot_utils")

SMALL_SIZE = 8
SMALLMED_SIZE = 15
MEDIUM_SIZE = 20
LARGE_SIZE = 28

def make_plot(x = None, y = None, f = 0, x_label = 'Time', y_label = 'Accelerations', model_name = '', title = '',plotname_suffix = '',text = '', out_dir = '.', is_cuda=False):

    if is_cuda:
        plt.figure(figsize=(15,15))
        plt.rc('font', size=LARGE_SIZE)
    else:
        plt.figure(figsize=(5,5))
        plt.rc('font', size=SMALL_SIZE)

    if x is not None:
        plt.plot(x, y, marker = '.', markersize=0.1, linewidth=0.6)
        plt.xlabel(x_label)
    else:
        plt.plot(y, marker = '.', markersize=0.1, linewidth=0.6, linestyle='None')
    plt.ylabel(y_label)
    plt.title(title)
    plot_name = get_plotname('acc', title)
    #plt.savefig('{0}/{1}.png'.format(out_dir, plot_name))
    plt.show()
    return

def make_lab_plot(x = None, y = None, f = 0, x_label = 'Time', y_label = 'Accelerations', model_name = '', title = '',plotname_suffix = '',text = '', out_dir = '.', is_cuda=False):

    if is_cuda:
        plt.figure(figsize=(15,15))
        plt.rc('font', size=LARGE_SIZE)
    else:
        plt.figure(figsize=(5,5))
        plt.rc('font', size=SMALL_SIZE)

    if x is not None:
        plt.scatter(x, y, marker = '.')
        plt.xlabel(x_label)
    else:
        plt.scatter(y, marker = '.')
    plt.ylabel(y_label)
    plt.gca().yaxis.set_ticks([0,1,2,3])
    plt.gca().yaxis.set_ticklabels(['Flat', 'Pothole', 'Patch', 'Crack'])
    plt.title(title)
    plot_name = get_plotname('acc', title)
    #plt.savefig('{0}/{1}.png'.format(out_dir, plot_name))
    plt.show()
    return

def get_plotname(prefix, plotname_suffix):
    out =  prefix + '_' + plotname_suffix
    return out

def plot_inputs(data, out_dir = '.'):
    accs = data.accs
    for inp in range(0, accs.shape[0]):
        acc = accs[inp,0,:]
        plt.figure()
        plt.plot(acc, label = 'Input acc, sample: {0}'.format(inp))
        plt.xlabel('Time')
        plt.ylabel('Accelation')
        #plt.xlim([0, plt.xlim()[1]])
        #plt.ylim([plt.ylim()[0]*0.8, plt.ylim()[1]*1.2])
        plt.legend()
        plot_name = 'input_'+str(inp)
        #plt.savefig('{0}/{1}.png'.format(out_dir, plot_name))
        plt.show()
    return


def learning_curve(x, y, y_label = 'Loss', model_name = '', plotname_suffix = '',text = '', out_dir = '.', is_cuda=False):
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


def plot_cm(cm, title_suffix='', plotname_suffix = '', out_dir = '.'):
    fig, ax = plt.subplots()
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='rocket_r', cbar=False, fmt='g'); #annot=True to annotate cells

    # Labels, title and ticks
    ax.set_title('Confusion Matrix: '+title_suffix)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(['Very Good', 'Good', 'Average', 'Poor'])
    ax.yaxis.set_ticklabels(['Very Good', 'Good', 'Average', 'Poor'])

    # Fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)

    # Save plot and show
    plot_name = get_plotname('CM_'+title_suffix, plotname_suffix)
    log.info('Saving as: {0}/{1}.png'.format(out_dir, plot_name))
    plt.savefig('{0}/{1}.png'.format(out_dir, plot_name))
    plt.show()

    # Save np
    np.savetxt('{0}/{1}.csv'.format(out_dir, plot_name), cm, fmt = '%i', header ='')

    return cm

def plot_tables(cm, f1_avg, f1_avg_w, title_suffix='', plotname_suffix = '', out_dir = '.'):

    precisions = []
    for col in range(0,  cm.shape[1]):
        col_data = cm[:, col]
        prec_this_class = float(col_data[col]/np.sum(col_data))
        precisions.append(prec_this_class)

    recalls = []
    for row in range(0,  cm.shape[0]):
        row_data = cm[row, :]
        recall_this_class = float(row_data[row]/np.sum(row_data))
        recalls.append(recall_this_class)

    f1s = []
    for p,r in zip(precisions, recalls):
        if (p+r)!=0:
            f1_this_class = 2*p*r/(p+r)
        else:
            f1_this_class = np.NaN
        f1s.append(f1_this_class)

    # Stack
    common = np.stack([precisions,recalls,f1s], axis=1)
    common = common*100
    common = np.round(common, decimals = 1)
    p = pd.DataFrame(data=common, index=['Flat', 'Pothole', 'Patch', 'Crack'], columns = ['Precision[%]', 'Recall[%]', 'F1-score[%]'])

    plot_name = get_plotname('summary_'+title_suffix, plotname_suffix)
    p.to_csv('{0}/{1}.csv'.format(out_dir, plot_name))
    return p
    '''
    fig, ax = plt.subplots()
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='rocket_r', cbar=False, fmt='g'); #annot=True to annotate cells

    # Labels, title and ticks
    ax.set_title('Confusion Matrix: '+title_suffix)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(['Flat', 'Pothole', 'Patch', 'Crack'])
    ax.yaxis.set_ticklabels(['Flat', 'Pothole', 'Patch', 'Crack'])

    # Fix for mpl bug that cuts off top/bottom of seaborn viz
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)

    # Save plot and show
    plot_name = get_plotname('CM_'+title_suffix, plotname_suffix)
    log.info('Saving as: {0}/{1}.png'.format(out_dir, plot_name))
    plt.savefig('{0}/{1}.png'.format(out_dir, plot_name))
    plt.show()

    # Save np
    np.savetxt('{0}/{1}.txt'.format(out_dir, plot_name), cm, fmt = '%i', header ='')
    '''
    return

def acc_curve(acc, valid_acc, y_label = 'Accuracy', model_name = '', plotname_suffix = '', text = '', out_dir = '.', setting='small'):
    if setting=='small':
        plt.figure(figsize=(5,5))
        plt.rc('font', size=SMALL_SIZE)
    elif setting=='large':
        plt.figure(figsize=(15,15))
        plt.rc('font', size=LARGE_SIZE)
    plt.plot(acc, label='Train Accuracy', color = 'b',  marker='.', markersize=2, linewidth = 0.9)
    plt.plot(valid_acc, label='Valid Accuracy', color = 'r',  marker='.', markersize=2, linewidth = 0.9)
    plt.title(model_name.replace('_',' ') + ' Model')
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    #plt.xlim([0, plt.xlim()[1]])
    plt.ylim([plt.ylim()[0], plt.ylim()[1]*1.2])
    plt.text(0.1, 0.9, text, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
    plt.legend()
    #plt.gca().xaxis.set_major_locator((MaxNLocator(integer=True)))
    plot_name = get_plotname('accuracy', plotname_suffix)
    plt.savefig('{0}/{1}.png'.format(out_dir, plot_name))
    plt.show()
    return

def combined(train_loss, valid_loss, train_acc, valid_acc, model_name = '', plotname_suffix = '', text = '', out_dir = '.'):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,15))
    plt.rc('font', size=MEDIUM_SIZE)
    ax1.plot(train_loss, label='Train Loss', color = 'b',  marker='.', markersize=6, linewidth = 0.9)
    ax1.plot(valid_loss, label='Valid Loss', color = 'r',  marker='.', markersize=6, linewidth = 0.9)
    ax1.set_title(model_name)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    #plt.xlim([0., plt.xlim()[1]])
    ax1.set_ylim([0, plt.ylim()[1]*1.2])
    ax1.legend()
    ax1.text(0.1, 0.1, text, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    ax2.plot(train_acc, label='Train acc', color = 'b',  marker='.', markersize=6, linewidth = 0.9)
    ax2.plot(valid_acc, label='Valid acc', color = 'r',  marker='.', markersize=6, linewidth = 0.9)
    ax2.set_title(model_name)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    #plt.xlim([0., plt.xlim()[1]])
    ax2.set_ylim([0, plt.ylim()[1]*1.2])
    ax2.legend()
    ax2.text(0.1, 0.1, text, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)

    plot_name = get_plotname('combined', plotname_suffix)
    #plt.tight_layout()
    plt.savefig('{0}/{1}.png'.format(out_dir, plot_name))
    plt.show()
    return

def plot_ts_windowcheck(data, title_suffix='', plotname_suffix = '', out_dir = '/'):
    true_labels = data['true_labels']
    time_window = data['mean_time_per_window']
    time = data['time']
    all_labels = data['all_labels']

    fig, ax = plt.subplots(figsize=(15,15))
    plt.rc('font', size=SMALLMED_SIZE)
    ax.scatter(time_window, true_labels, marker = 's',s = 10, c='blue', label = 'True per window')
    ax.scatter(time, all_labels, alpha=0.5, marker = 'o', s = 3, c='red', label = 'All true')
    ax.set_yticks([0,1,2,3,5,10,15])
    plt.ylim([0,4])
    ax.set_title(title_suffix)
    ax.set_xlabel('Time')
    ax.set_ylabel('Label')
    ax.legend()
    plot_name = get_plotname(title_suffix, plotname_suffix)
    plt.savefig('{0}/windowcheck_{1}.png'.format(out_dir, plot_name))
    plt.show()
    return


def plot_ts_scatter(data, title_suffix='', plotname_suffix = '', out_dir = '/'):
    true_labels = data['true_labels']
    pred_labels = data['pred_labels']
    time_window = data['mean_time_per_window']

    fig, ax = plt.subplots(figsize=(15,15))
    plt.rc('font', size=SMALLMED_SIZE)
    ax.scatter(time_window, true_labels, marker = 's',s = 10, c='blue', label = 'True')
    ax.scatter(time_window, pred_labels, alpha=0.5, marker = 'o', s = 3, c='red', label = 'Predicted')
    ax.set_yticks([0,1,2,3,5,10,15])
    plt.ylim([0,4])
    ax.set_title(title_suffix)
    ax.set_xlabel('Time')
    ax.set_ylabel('Label')
    ax.legend()
    plot_name = get_plotname(title_suffix, plotname_suffix)
    plt.savefig('{0}/scatter_{1}.png'.format(out_dir, plot_name))
    plt.show()
    return


def plot_ts(data, title_suffix='', plotname_suffix = '', out_dir = '/', is_cuda = False):
    true_labels = data['true_labels']
    pred_labels = data['pred_labels']
    time_window = data['mean_time_per_window']
    time = data['time']
    acc = data['acc']
    all_labels = data['all_labels']

    if is_cuda:
        fig, ax1 = plt.subplots(figsize=(25,25))
        plt.rc('font', size=LARGE_SIZE)
    else:
        fig, ax1 = plt.subplots(figsize=(15,15))
        plt.rc('font', size=SMALLMED_SIZE)

    l1 = ax1.plot(time_window, true_labels, linestyle="None", marker = 's', markersize = 4, mfc='blue', mec = 'blue', label = 'True')
    l2 = ax1.plot(time_window, pred_labels, linestyle="None", alpha=0.5, marker = 'o', markersize = 2, mfc='red', mec='red', label = 'Predicted')
    ax1.set_yticks([0,1,2,3])
    ax1.set_ylim([ax1.get_ylim()[0], ax1.get_ylim()[1]*1.2])
    ax1.set_ylabel('Label')
    ax1.yaxis.set_ticklabels(['Flat', 'Pothole', 'Patch', 'Crack'])
    ax1.set_title(title_suffix)

    ax2 = ax1.twinx()
    l3 = ax2.plot(time, acc, linewidth = 1, c = 'yellow', alpha=0.9, label = 'Acceleration')
    ax2.set_ylim([ax2.get_ylim()[0], ax2.get_ylim()[1]*1.2])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Acceleration')

    l_all = l1+l2+l3
    labs = [l.get_label() for l in l_all]
    ax2.legend(l_all, labs, loc='upper right')

    plot_name = get_plotname(title_suffix, plotname_suffix)
    fig.tight_layout()
    plt.savefig('{0}/TS_{1}.png'.format(out_dir, plot_name))
    plt.show()
    return


def plot_times(data, min_ind = 0 , max_ind = -1):
#def plot_times(data, min_ind = 9000 , max_ind = 12000):
    max_ind=max_ind
    t = data.times[:,:,min_ind:max_ind]
    a = data.accs[:,:,min_ind:max_ind]
    l = data.labels[:,:,min_ind:max_ind]
    le = len(np.where(l!=0)[2])-1
    #le = None
    fig, ax1 = plt.subplots()
    ax1.set_title('Combined: 1st defect')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Acc')
    ax1.plot(t.reshape(-1),a.reshape(-1))
    ax1.set_ylim((-30,30))
    ax2 = ax1.twinx()
    ax2.set_ylabel('True label')
    ax2.plot(t.reshape(-1),l.reshape(-1), c = 'red', linestyle='None', marker='.', markersize='0.7')
    #ax2.yaxis.set_ticklabels(['Flat', 'Pothole', 'Patch', 'Crack'])
    ax2.set_ylim([ax1.get_ylim()[0], ax1.get_ylim()[1]])
    plt.show()
    return t,a,l,le

def plot_times_a(t,a,l, title=''):
    #le = len(np.where(l!=0)[2])-1
    le = None
    fig, ax1 = plt.subplots()
    ax1.set_title('Patch: '+ title)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Acc')
    #ax1.set_ylim((-30,30))
    ax1.scatter(t.reshape(-1),a.reshape(-1), s=1)
    ax2 = ax1.twinx()
    ax2.set_ylabel('True label')
    ax2.plot(t.reshape(-1),l.reshape(-1), c = 'red', linestyle='None', marker='.', markersize='0.7')
    #ax2.yaxis.set_ticklabels(['Flat', 'Pothole', 'Patch', 'Crack'])
    ax2.set_ylim([-5,5])
    #ax2.set_ylim([ax1.get_ylim()[0], ax1.get_ylim()[1]])
    plt.show()
    return t,a,l,le

