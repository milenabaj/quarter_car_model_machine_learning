"""
A script to apply the sliding window approach on input time series data. Creates fully prepared datasets for analysis.

@author: Milena Bajic (DTU Compute)
"""


import sys,os, glob, time
import subprocess
import argparse
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Window_dataset():

    def __init__(self, input_dir, filestring, win_size = 2, out_dir = '', is_test = False, scale_speed = True):

        # Initial processing time
        t0=time.time()

        # Get from input
        self.input_dir = input_dir
        self.out_dir = out_dir
        self.filestring = filestring
        self.win_size = win_size
        self.test = is_test
        self.scale_speed = scale_speed

        # Create output dir for this filetype 
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            
        # Load pickle
        self.input_dataframe = self.load_pickle(input_dir, filestring)

        # Remove rows with 0 points recorded, n_points[s] = 3.6*fs*defect_width/v[km/h]
        print('Is test: {0}'.format(is_test))
        if self.test:
            self.input_dataframe = self.remove_samples_with_zero_counts(self.input_dataframe).head(100)
            self.n_split_rows_length = 20
        else:
            self.input_dataframe = self.remove_samples_with_zero_counts(self.input_dataframe)
            self.n_split_rows_length = 1000

        # Take only needed columns
        self.input_columns = ['time','distance','speed', 'acceleration', 'severity', 'type', 'defect_width', 'defect_height']
        self.deciding_column = 'type'
        self.filestring = self.filestring
        self.input_dataframe = self.input_dataframe[self.input_columns]

        # Scale speed
        if self.scale_speed:
            print('Scaling speed')
            scaler_file = self.out_dir+'/train_scaler_speed.pt'
            speed = self.input_dataframe['speed'].to_numpy()
            speed = speed.reshape(-1,1)
            print(speed[0], speed.shape)
            if self.filestring == 'train':
                scaler = MinMaxScaler().fit(speed)
                pickle.dump(scaler, open(scaler_file, 'wb'))
            else:
                scaler = pickle.load(open(scaler_file, 'rb'))
            self.input_dataframe['scaled_speed'] = scaler.transform(speed)
            print(self.input_dataframe['speed'][0], self.input_dataframe['speed'].shape)
            print(self.input_dataframe['scaled_speed'][0], self.input_dataframe['scaled_speed'].shape)
            
        sys.exit(0)
        # Window columns to save
        self.window_columns = [col for col in self.input_columns if col!=('distance')]
        self.window_columns.append('window_class')
        
        # Split a very large input df into smaller ones to fit into RAM more easily
        self.n_input_rows = self.input_dataframe.shape[0]
        self.last_split = int(self.n_input_rows/self.n_split_rows_length)
        self.index_list =  [n*self.n_split_rows_length for n in range(1,self.last_split+1)]
        self.split_input_dataframes = np.split(self.input_dataframe, self.index_list)
        self.n_splits = len(self.split_input_dataframes)
        print('Number of split dataframes: {0}'.format(self.n_splits))

        for df_i, df in list(enumerate(self.split_input_dataframes)):
            print('===> Passing df: ',df_i)
            df.reset_index(inplace=True, drop=True)
            self.make_sliding_window_df(df_i, df)
        
        dt = round(time.time()-t0,1)
        print('Time to process: {0} s'.format(dt))

    def load_pickle(self, input_dir, string):
        filename = '{0}/{1}_scaled.pkl'.format(input_dir, string)
        print('Loading: {0}'.format(filename))
        with open(filename, "rb") as f:
            df = pickle.load(f)
        return df

    def remove_samples_with_zero_counts(self, input_dataframe):
        # Remove samples with too narrow defects so there is no point "caught" in type and severity
        input_dataframe['keep'] = input_dataframe.type.apply(lambda row: np.count_nonzero(row)>0)
        input_dataframe = input_dataframe[ input_dataframe['keep']==True ]
        input_dataframe.drop(['keep'],axis=1, inplace = True)
        input_dataframe.reset_index(drop=True, inplace=True)
        return input_dataframe


    def make_sliding_window_df(self, df_i, input_dataframe_part):
        # Making sliding window (each window: constant in distance, variable length, slide by 1 point)
        print('Making sliding window')
        window_df = pd.DataFrame([], columns = self.window_columns)

        # Fill Dataframe with windows from initial one
        for index, row in input_dataframe_part.iterrows():
            if (index%500==0):
                print('Processing row: {0}/{1}'.format(index,input_dataframe_part.shape[0]))
            row_df = self.make_sliding_window_row(row)
            window_df = window_df.append(row_df)

        window_df.reset_index(inplace=True, drop=True)

        # Save picklescree
        self.save_pickle(window_df, self.out_dir, self.filestring+'_'+ str(df_i))
        return

    def make_sliding_window_row(self, row):
        row_df = pd.DataFrame([], columns = self.window_columns)

        end_index = np.where(row.distance > 100 - self.win_size )[0][0]-1 #
        #print(end_index, row.distance[end_index])   # end index in the whole row (so the last sample is 2m)

        # Loop over the windows
        for i in range(0, end_index+1):
            try:
                # Get min and max index of this window
                window_start_meters= row.distance[i]
                window_end_meters= window_start_meters + self.win_size
                window_end_index = np.where(row.distance>window_end_meters)[0][0]
                #print(i, window_end_index, window_start_meters, window_end_meters)

                # If the window is fully flat, add with a small prob. equal to how probable is each defect
                window_is_flat = np.all(row[self.deciding_column][i: window_end_index]==0)
                remove_window = False
                if window_is_flat:
                   remove_window = random.randrange(100)<99  # keep with 2% probability
                if remove_window:
                    continue

                # Put this window into row df data
                for col in self.window_columns:
                    if col=='window_class': # compute window class column
                        unique_classes = np.unique(row['type'][i: window_end_index]) #possible are only 1-label windows or windows with 0 (no defect) and 1 defect
                        if len(unique_classes)==1:
                           row_df.at[i,col] = unique_classes[0]
                        elif len(unique_classes)==2:
                            row_df.at[i,col] = list(filter(lambda c: c!=0, unique_classes ))[0]
                        else:
                            raise Error("More than 1 defect per window not implemented.")
                    elif isinstance(row[col],np.ndarray): # fill numpy array columns
                        row_df.at[i,col] = row[col][i: window_end_index]
                    else:
                        row_df.at[i,col] = row[col] #float or string, just repeat

            except:
                pass
        return row_df

    def save_pickle(self, df, out_dir, df_type):
        print('Saving {0} as pickle.'.format(df_type))
        pickle_name = out_dir+'/'+df_type+'_windows.pkl'
        df.to_pickle(pickle_name)
        print('Wrote output file to: ',pickle_name)
        return



#===============================#
# ============================= #

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Please provide command line arguments.')
    parser.add_argument('--is_test', action='store_true', 
                        help = 'If test is true, will process 100 rows only (use for testing purposes).') #store_true sets default to False 
    parser.add_argument('--window-size', default = 5, type=int,
                        help = 'Window size.')
    #parser.add_argument('--input_dir', default = '/Users/mibaj/quarter_car_model_machine_learning/data/Golden-car-simulation-August-2020/train-val-test-normalized-split-into-windows-cluster',
    #                   help = 'Input directory.')
    #parser.add_argument('--output_dir_base', default = '/Users/mibaj/quarter_car_model_machine_learning/data/Golden-car-simulation-August-2020',
    #                    help='Directory base where a new directory with output files will be created.')
    parser.add_argument('--input_dir', default = '/dtu-compute/mibaj/Golden-car-simulation-August-2020/train-val-test-normalized',
                        help = 'Input directory.')
    parser.add_argument('--output_dir_base', default = '/dtu-compute/mibaj/Golden-car-simulation-August-2020',
                        help='Directory base where a new directory with output files will be created.')

    # Parse arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir_base
    is_test = args.is_test
    window_size = args.window_size
    
    # Print configuration
    print('Window_size: {0}'.format(window_size))
    print('Is test: {0}'.format(is_test))
    
    for filetype in ['train','valid','test']:
        print('Processing: {0}'.format(filetype))
        
        # Make output directory
        out_dir = '{0}/train-val-test-normalized-split-into-windows-size-{1}'.format(output_dir, window_size)
        print('Output directory: {0}'.format(out_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    
        # Process
        # ======#
        result = Window_dataset(input_dir, filetype, win_size = window_size, out_dir = out_dir + '/'+str(filetype), is_test = is_test)

#
# TODO: scale speeds
# Recreate all