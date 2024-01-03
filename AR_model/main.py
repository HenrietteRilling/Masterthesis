# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:09:19 2023

@author: Henriette
"""

import json
import csv

from utils import load_data
from run_sim_LSTM import run_LSTM


if __name__ == '__main__':
    
    #read configuration
    with open('run_config_LSTM.json', 'r', encoding='utf-8') as f:
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
                            #run simulation for current configuration
                            run_LSTM(X, test_id, respath,
                                      config['train_period'], config['val_period'], config['test_period'],
                                      config['training_horizon'], config['imputation_horizon'], 
                                      config['epochs'], batch_size, config['n_models'],
                                      neurons, nr_layers, window
                                      )
                            configs_tested.append([respath, model, window, batch_size, neurons, nr_layers, test_id, config['prcp_station']])
                            print('Finished config')


    #Save configs in csv file
    csv_file_path = './Results/configs.csv'    
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        [csv_writer.writerow(row) for row in configs_tested]



    

