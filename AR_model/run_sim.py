# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:09:19 2023

@author: Henriette
"""

import os
import numpy as np
import pandas as pd

import torch
import matplotlib.pyplot as plt
#Set the number of CPU threads that PyTorch will use for parallel processing
torch.set_num_threads(8)

from datetime import datetime
from utils import scale_data, timeseries_dataset_from_array, get_dataloader, get_WL_data, get_prcp_data, get_test_data
from plot_utils import plot_losses
from model import sampleFFNN_AR
from trainer import Trainer
from metrics import rmse, PI
from bokeh.plotting import figure, show, output_file
from bokeh.models import DatetimeTickFormatter




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
test_station='Bjerringbro'
test_id=station_name_to_id.get(test_station)
test_prcp='05135'

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

#scale and normalise such that all data has a range between [0,1], store scaler for rescaling
X_train_sc, train_sc = scale_data(X_train)
X_val_sc = train_sc.transform(X_val)
X_test_sc = train_sc.transform(X_test)

#get scaler only for waterlevel for unscaling predictions
_, train_WL_sc=scale_data(X_train[[test_id]])



#choose horizons that are tested
forecast_horizons=[1, 12, 24, 48, 168]  #1h, 1/2d, 1d, 2d, 1 week
test_horizons=[1, 24, 48, 168,  672] #1h, 1d, 2d, 1 week, 1 month
nr_of_model_runs=5
epochs=3
batch_size=100 #number of batches that is processed at once 


for fh in forecast_horizons[:2]:

    respath=f'./Results/F1_FH_{fh}'
    if not os.path.exists(respath): os.makedirs(respath)
    for i in range(nr_of_model_runs):
        print(f'\nModel {i} of forecast horizon {fh}') 
        


        #delete any existing losslog/files, to only save losses of current model run
        losslogpath=os.path.join(respath, f'losslog_{i}.csv')
        if os.path.exists(losslogpath): os.remove(losslogpath)
        
        #set forecast horizon
        forecast_horizon=fh #for how many timesteps should the model predict in total?
        horizon=1 #how many time steps are predicted at each time step, we try to make one-step predictions
        ###################### Batch data ##########################################
        #get input in batches with w timesteps 
        features_train, labels_train = timeseries_dataset_from_array(X_train_sc, forecast_horizon, horizon, AR=True)
        features_val, labels_val=timeseries_dataset_from_array(X_val_sc, forecast_horizon, horizon, AR=True) 
        
        #get data_loader for all data, data_loader is an torch iterable to be able to iterate over batches
        dataset_train, data_loader_train = get_dataloader(features_train, labels_train, batch_size=batch_size)
        dataset_val, data_loader_val=get_dataloader(features_val, labels_val, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???
        
        #############################################################
        #set up an autregressive model - the autoregressive model loop is defined in AR_model.py. The class in there imports a neural network configuration that is defined inside AR_nn_models.py, a feed forward model with 2 layers and as many neurons as defined in hidden size
        input_size=features_train.shape[-1]  #number of input features 
        hidden_size=25 #number of neurons in hidden layer
        output_size=horizon 
        
        model=sampleFFNN_AR(input_size, hidden_size, output_size) 
 
        trainer = Trainer(model,epochs,respath, batch_size, i)
        trainer.fit(data_loader_train,data_loader_val)
        ###################
        
        #plot losses
        plot_losses(losslogpath, os.path.join(respath, f'losses_{i}.png'))
        
        # ###################################################################
        # ###################### Testing
        print(f'\nDone fitting model {i}')
        
        #load weights of best model
        model.load_state_dict(torch.load(os.path.join(respath,'weights_{i}.pth')))
        model.eval()
        
        for th in test_horizons[:2]:
            print(f'Testing for imputation of {th} hours')
            metricspath=os.path.join(respath, f'metrics_{th}.txt')
            #create test features and label
            features_test, labels_test =timeseries_dataset_from_array(X_test_sc, th, horizon, AR=True)
            dataset_test, data_loader_test=get_dataloader(features_test, labels_test, batch_size=batch_size, shuffle=False)
        
            #generate predictions based on test set
            all_test_preds=[]
            for step, (x_batch_test, y_batch_test) in enumerate(data_loader_test):
                #generate prediction
                pred=model(x_batch_test)
                #calculate metric for each batch, exlcude precipitation in target
                all_test_preds.append(pred)
            
            ####################################
            #Metrics               
            #concat list elements along "time axis" 0
            preds_test=torch.cat(all_test_preds, 0).detach().numpy()
            #unscale all predicitons and labels
            preds_test_unsc=train_WL_sc.inverse_transform(preds_test[:,:,0])
            labels_test_unsc=train_WL_sc.inverse_transform(labels_test[:, :,0]) #include only water level
            features_test_unsc=train_WL_sc.inverse_transform(features_test[:,:,0]) #needed for PI, inlcude only water level
            #calculate Metrics
            rmse(labels_test_unsc[:len(preds_test),:], preds_test_unsc, savepath=metricspath)
            PI(labels_test_unsc[:len(preds_test),:], features_test_unsc[:len(preds_test),:] ,preds_test_unsc, savepath=metricspath)


            #extract first time step of each data window
            #there are no predictions for the last timesteps in the length of the forecast horizon-1, as there's no target to compare to
            plot_test=np.concatenate(([np.nan], preds_test_unsc[:,0], np.full(forecast_horizon-1, np.nan)))
        
        
            # Create a Bokeh figure
            p = figure(x_axis_label='Date', y_axis_label='Water level [m]', title=f'FH: {fh} TH: {th}')
            
            # Plot prediction
            p.line(pd.to_datetime(X_test.index), plot_test, line_width=2, legend_label='Prediction', line_color='blue')
            
            # Plot observation
            p.line(pd.to_datetime(X_test.index), X_test[test_id], line_width=2, legend_label='Observation', line_color='green')
            
            # Customize the plot
            p.legend.location = 'top_left'
            p.xaxis.formatter = DatetimeTickFormatter(days=["%Y-%m-%d"])
            
            # Save the plot
            output_file(os.path.join(respath, f'preds_test_{i}_{th}.html'))
            show(p)
            # plt.figure()   
            # plt.plot(pd.to_datetime(X_test.index), plot_test, label='prediction')
            # plt.plot(pd.to_datetime(X_test.index), X_test[test_id], label='observation')
            # plt.xlabel('Date')
            # plt.ylabel('Water level [m]')
            # plt.legend()  
            # plt.title(f'Model: {model_name}')
            # plt.show()
            # plt.savefig(os.path.join(respath, 'preds_test.png'))
            # # plt.close()    
