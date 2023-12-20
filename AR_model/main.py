# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:09:19 2023

@author: Henriette
"""
import os
import sys
import shutil
import time
import json
import ast

import numpy as np
import pandas as pd

from utils import get_WL_data, get_prcp_data, get_test_data
from run_sim_LSTM import run_LSTM


def load_data(WL_stat, prcp_stat):
    #############################################
    #load data
    WL, _, station_name_to_id, _ = get_WL_data(r'./Data')
    prcp=get_prcp_data(r'./Data', r'./Data', join=True)

    #remove outliers from WL data with z-score in order to train model with "good" time-series
    #calculate z-score
    zscore=(WL-WL.mean())/WL.std()
    #threshhold for detecting outliers
    threshold=3
    WL_wo_anom= WL 
    for col in WL.columns:
        WL_wo_anom[col][np.abs(zscore[col])>threshold]=np.nan

    #select test stations and extract data
    test_station=WL_stat
    test_id=station_name_to_id.get(test_station)
    test_prcp=prcp_stat

    X_WL=get_test_data(test_id, WL_wo_anom)
    X_prcp=get_test_data(test_prcp, prcp)

    #merge precipitation and WL data, select overlapping timeperiod
    X=pd.concat([X_WL, X_prcp], axis=1).loc[X_WL.index.intersection(X_prcp.index)]
    # X=X_WL
    return X, test_id




if __name__ == '__main__':
    
    #read configuration
    with open('run_config_LSTM.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    respath_list=[]
    for model in config['models']:
        #load data
        X, test_id=load_data(config['test_station'], config['prcp_station'])
        if model=='lstm':
            for window in config['window_size']:
                for batch_size in config['batch_size']:
                    for neurons in config['neurons']:
                        for nr_layers in config['layers']:
                                                       
                            respath=f'./Results/{model}_W_{window}_B_{batch_size}_N_{neurons}_L_{nr_layers}'
                            run_LSTM(X, test_id, respath,
                                     config['train_period'], config['val_period'], config['test_period'],
                                     config['training_horizon'], config['imputation_horizon'], 
                                     config['epochs'], batch_size, config['n_models'],
                                     neurons, nr_layers, window
                                     )
                            respath_list.append(respath)
    
    
    

