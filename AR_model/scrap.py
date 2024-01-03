# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:10:41 2023

@author: Henriette
"""


import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime
from utils import get_WL_data, get_prcp_data, get_test_data
from plot_utils import cm2inch


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
test_station='ns Uldumkær'
test_id=station_name_to_id.get(test_station)
test_prcp='05225'

X_WL=get_test_data(test_id, WL_wo_anom)
X_prcp=get_test_data(test_prcp, prcp)

#merge precipitation and WL data, select overlapping timeperiod
X=pd.concat([X_WL, X_prcp], axis=1).loc[X_WL.index.intersection(X_prcp.index)]
# X=X_WL 

#################################################
#split data in 70% train, 20% val, 10% test
#note: pd slicing is inlcusive
X_train=X['2012-01-01':'2018-12-31']
X_val=X['2019-01-01':'2021-12-31']
X_test=X['2022-01-01':'2022-12-31']


fig, ax1 = plt.subplots()

# Plot water level on the bottom axis
ax1.plot(X[test_id], color='blue', label='Water Level')
ax1.set_xlabel('Date')
ax1.set_ylabel('Water Level', color='blue')
ax1.tick_params('y', colors='blue')
ax1.axhline(y=49.7, color='red')
# Create a second y-axis for precipitation
ax2 = ax1.twinx()
ax2.plot(-X[test_prcp], color='green', label='Precipitation')
ax2.set_ylabel('Precipitation', color='green')
ax2.tick_params('y', colors='green')
ax2.set_ylim(-20, 0)
# Invert the tick labels on the second y-axis such that they are displayed positive

yticks = ax2.get_yticks()
ax2.set_yticks(yticks)
ax2.set_yticklabels([abs(y) for y in yticks])



'''
Old function for plotting directly in simulation
'''
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


    # for i, conf in enumerate(configs_tested):
    #     weight_paths=get_best_weigth_paths(conf[0],config['n_models'])
    #     plot_imputation(X, test_id, config['prcp_station'], conf[0], config['train_period'], config['test_period'], config['plot_horizon'], 
    #                     conf[3], conf[4], conf[5], conf[2], weight_paths) 
 