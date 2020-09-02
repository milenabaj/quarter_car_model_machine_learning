"""
A script to prepare full/train/valid/test pickle files with Pandas dataframe containing simulation data for car traversing a road with
crack/patch/pothole defects, with various car speeds and defect geometries.

@author: Milena Bajic (DTU Compute)
"""


import sys,os, glob
import pickle
from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import argparse

def save_split_df(df, df_type, out_dir):
    pickle_name = out_dir+'/'+df_type+'.pkl'
    df.to_pickle(pickle_name)
    print('Wrote output file to: ',pickle_name)
    return


# ============================= #
# ============================= #

if __name__ == "__main__":

    home = os.path.expanduser('~')
    parser = argparse.ArgumentParser(description='Please provide command line arguments.')
    parser.add_argument('--input_dir', default = '{0}/quarter_car_model_data_preparation/data/Golden-car-simulation-August-2020/Matlab-Files'.format(home),
                        help = 'Input directory containing single-defect .mat files.')
    parser.add_argument('--output_dir_base', default = '{0}/quarter_car_model_data_preparation/data/Golden-car-simulation-August-2020'.format(home),
                        help='Directory base where a new directory with output files will be created.')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir_base = args.output_dir_base

    # Make output directory
    out_dir = '{0}/train-val-test'.format(output_dir_base)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    # === CONVERT MAT FILES TO ONE PANDAS DATAFRAME AND SAVE === #
    # ========================================================== #

    # Load mat files
    file_pattern = '{0}/*.mat'.format(input_dir)
    df = pd.DataFrame(columns = ['severity', 'type', 'time', 'distance', 'acceleration']) #others are added with append
    n_files = glob.glob(file_pattern)
    itr = 0
    for filename in glob.glob(file_pattern):
         f = loadmat(filename) # loaded as python dictionary
         if itr%1000==0:
             print('N processed files: {0}, currently loaded: {1}'.format(itr, filename))

         remove_keys = ['__header__', '__version__', '__globals__']
         for key in remove_keys:
             del f[key]

         for key in f:
             f[key] = [ f[key].reshape(-1) ]

         # Df for this file
         df_tmp = pd.DataFrame.from_dict(f)

         # Add info from file name to the dataframe
         file_info =  filename.split('/')[-1].split('.mat')[0]

         # Add info from filename to df
         df_tmp['defect_label'] = file_info.split('qcar_AccZ_')[1].split('_')[0]
         df_tmp['defect_width'] = int(file_info.split('width')[1].split('_mm')[0])
         df_tmp['defect_height'] = int(file_info.split('depth')[1].split('_mm')[0])
         df_tmp['speed'] = int(file_info.split('speed')[1].split('_kmh')[0])
         df_tmp['sampling_freq'] = int(file_info.split('rate')[1].split('_Hz')[0])

         # Save full filename too
         df_tmp['filename'] = filename.split('/')[-1]

         # Append dataframe for this file to the combined one
         df = df.append(df_tmp) # each file data is now a row

         itr = itr + 1

    # Reset index of the final dataframe
    df.reset_index(inplace=True, drop=True)

    # Dump dataframe to a pkl file
    pickle_name = '{0}/simulation_full.pkl'.format(out_dir)
    with open(pickle_name, 'wb') as outfile:
         pickle.dump(df, outfile, pickle.HIGHEST_PROTOCOL)
         print('Wrote output file to: ',pickle_name)


    # === SPLIT DATAFRAME INTO TRAIN/VALID/TEST AND SAVE === #
    # ====================================================== #
    test_size=0.2
    val_size=0.2
    trainval, test = train_test_split(df, test_size=test_size, random_state=11, shuffle=True)
    train, val = train_test_split(trainval,  test_size=val_size/(1-test_size), random_state=11)

    train.reset_index(inplace=True, drop=True)
    val.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    save_split_df(train, 'train', out_dir)
    save_split_df(val, 'valid', out_dir)
    save_split_df(test, 'test', out_dir)






