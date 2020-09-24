#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:49:57 2020

@author: milena
"""

import sys,os, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_filename = '/Users/milena/quarter_car_model_machine_learning/data/Golden-car-simulation-August-2020/train-val-test-normalized-split-into-windows-size-5/train/train_0_windows.pkl'
with open(train_filename,'rb') as f:
    df = pickle.load(f)
    
h = df.defect_height.value_counts()
h.sort_index(inplace=True)

w = df.defect_width.value_counts()
w.sort_values(inplace=True)

df.speed.min()