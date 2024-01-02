# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:09:19 2023

@author: Henriette
"""
import os
import glob
import json
import csv

import numpy as np
import pandas as pd
import re

from utils import get_WL_data, get_prcp_data, get_test_data
from run_sim_LSTM import run_LSTM
from plot_imputation import plot_imputation


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


def get_best_weigth_paths(path, nr_of_models):
    '''
    The same model configuration is trained for n times for balancing statistical variations. 
    Visualisation of results is based on model with lowest validation error, this function identifies
    the model with the lowest valdiation error among the model runs.
    '''
    losslogpaths=glob.glob(os.path.join(path, '*losslog*.csv'))
    #define regular expression pattern for extracting training horizon of weight file name
    pattern=re.compile(r'losslog_(\d+)_\d+\.csv')
    #sort weights from shortest to longest training horizon
    losslogpaths=sorted(losslogpaths, key=lambda x: int(re.search(pattern, x).group(1)))
    best_model_paths=[]
    #loop over lossfiles and find for each training horizon the model with the lowest validation error
    for i, losspath in enumerate(losslogpaths):
        #read very last validation loss from file
        if i%nr_of_models==0:
            val_loss=1.0
            best_model_paths.append(losspath)
        
        #read only value of last row as this corresponds to the lowest validation error
        cur_val_loss=pd.read_csv(losspath, header=None, sep=';',usecols=[1],skiprows=lambda x: x < sum(1 for line in open(losspath)) - 1).iloc[0,0]
        if cur_val_loss<val_loss:
            val_loss=cur_val_loss
            #replace loss-expressions with pattern of weightstring
            best_model_paths[-1]=losspath.replace('losslog', 'weights').replace('csv', 'pth')    
                
    return best_model_paths
    




if __name__ == '__main__':
    
    #read configuration
    with open('run_config_LSTM.json2', 'r', encoding='utf-8') as f:
        config = json.load(f)
    # respath_list=[r'./Results/lstm_W_10_B_100_N_25_L_1']
    configs_tested=[]
    #train models
    for model in config['models']:
        #load data
        X, test_id=load_data(config['test_station'], config['prcp_station'])
        if model=='lstm':
            for window in config['window_size']:
                for batch_size in config['batch_size']:
                    for neurons in config['neurons']:
                        for nr_layers in config['layers']:
                                                       
                            respath=f'./Results/{model}_W_{window}_B_{batch_size}_N_{neurons}_L_{nr_layers}'
                            print(f'\nStart config: W {window}, B {batch_size}, N {neurons}, L {nr_layers}')
                            run_LSTM(X, test_id, respath,
                                      config['train_period'], config['val_period'], config['test_period'],
                                      config['training_horizon'], config['imputation_horizon'], 
                                      config['epochs'], batch_size, config['n_models'],
                                      neurons, nr_layers, window
                                      )
                            configs_tested.append([respath, model, window, batch_size, neurons, nr_layers])
                            print('Finished config')


    for i, conf in enumerate(configs_tested):
        weight_paths=get_best_weigth_paths(conf[0],config['n_models'])
        plot_imputation(X, test_id, config['prcp_station'], conf[0], config['train_period'], config['test_period'], config['plot_horizon'], 
                        conf[3], conf[4], conf[5], conf[2], weight_paths) 
 

    csv_file_path = './Results/configs.csv'    
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        [csv_writer.writerow(row) for row in configs_tested]



    

