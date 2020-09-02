"""
A scaling script to be run on prepared train/val/test data.

@author: Milena Bajic (DTU Compute)
"""

import sys,os, glob
import argparse
import pickle
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def save_split_df(df, df_type, out_dir):
    print('Saving {0} as pickle.'.format(df_type))
    pickle_name = out_dir+'/'+df_type+'.pkl'
    df.to_pickle(pickle_name)
    print('Wrote output file to: ',pickle_name)
    return df

def load_pickle(input_dir, string):
    filename = '{0}/{1}.pkl'.format(input_dir, string)
    print('Loading: {0}'.format(filename))
    with open(filename, "rb") as f:
        df = pickle.load(f)
    return df

def scale_train_df(train_df, cols_to_scale = [' Hastighed [m/s]', ' Acceleration [m/s²]']):
    print('Scaling train data')

    # Get min and max for full dataset
    train_df_scaled = pd.DataFrame([], columns = train_df.columns)
    train_df_maxmin = pd.DataFrame([], columns = train_df.columns)

    for col in train_df.columns:
       if col in cols_to_scale:
           # Get parameters
           col_min = min(train_df[col].apply(lambda row: row.min()))
           col_max = max(train_df[col].apply(lambda row: row.max()))
           col_diff = col_max - col_min

           # Scale
           train_df_scaled[col] = train_df[col].apply(lambda row: (row-col_min)/col_diff)

           # Save scaler info
           train_df_maxmin.at[0,col] = col_min
           train_df_maxmin.at[1,col] = col_max
           train_df_maxmin.at[2,col] = col_diff #difference
       else:
            train_df_scaled[col] = train_df[col]

    return train_df_scaled,  train_df_maxmin


def scale_non_train_df(df, train_df_maxmin):
    print('Scaling valid/test data')

    df_scaled = pd.DataFrame([], columns = df.columns)

    for col in df.columns:
        # Get params
        col_min = train_df_maxmin.at[0,col]
        col_max = train_df_maxmin.at[1,col]
        col_diff = train_df_maxmin.at[2,col] #difference

        # Scale
        if col_min is np.nan:
            df_scaled[col] = df[col]
        else:
            df_scaled[col] = df[col].apply(lambda row: (row-col_min)/col_diff)

    return df_scaled


# ============================= #
# ============================= #
if __name__ == "__main__":

    home = os.path.expanduser('~')
    parser = argparse.ArgumentParser(description='Please provide command line arguments.')
    parser.add_argument('--input_dir', default = '{0}/quarter_car_model_data_preparation/data/Golden-car-simulation-August-2020/train-val-test'.format(home),
                        help = 'Input directory containing single-defect .mat files.')
    parser.add_argument('--output_dir_base', default = '{0}/quarter_car_model_data_preparation/data/Golden-car-simulation-August-2020'.format(home),
                        help='Directory base where a new directory with output files will be created.')

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir_base = args.output_dir_base

    # Make output directory
    out_dir = '{0}/train-val-test-normalized'.format(output_dir_base)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load files
    train = load_pickle(input_dir, 'train')
    valid = load_pickle(input_dir, 'valid')
    test = load_pickle(input_dir, 'test')

    # Scale
    train_scaled, scaler = scale_train_df(train, cols_to_scale = ['acceleration'])
    valid_scaled = scale_non_train_df(valid,scaler)
    test_scaled = scale_non_train_df(test, scaler)

    # Save
    save_split_df(train_scaled, 'train_scaled', out_dir)
    save_split_df(valid_scaled, 'valid_scaled', out_dir)
    save_split_df(test_scaled, 'test_scaled', out_dir)
    save_split_df(scaler, 'scaler', out_dir)