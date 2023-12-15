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


#############################
#choose which models are tested, options: ['FFNN_AR','FFNN_AR1','FFNN_AR2']
model_tested='FFNN_AR'

# =============================================================================
# FFNN_AR: autoregressive model, creating a prediciton of length forecast_window
# by making one-step predictions on each timestep
# =============================================================================
import pdb
pdb.set_trace()
if  model_tested =='FFNN_AR':
    model_name=model_tested
    print(f'\nRunning {model_name}')
    respath=f'./results/{model_name}'
    if not os.path.exists(respath): os.makedirs(respath)
    #delete any existing losslog/files, to only save losses of current model run
    losslogpath=os.path.join(respath, 'losslog.csv')
    if os.path.exists(losslogpath): os.remove(losslogpath)
    metricspath=os.path.join(respath, 'metrics.txt')
    if os.path.exists(metricspath): os.remove(metricspath)
    
    forecast_horizon=20 #for how many timesteps should the model predict in total?
    horizon=1 #how many time steps are predicted at each time step, we try to make one-step predictions
    epochs=150
    batch_size=100 #number of batches that is processed at once 
    
    #batch data
    #get input in batches with w timesteps 
    #we feed predictions back in case of gaps, therefore we need a to extend the target by the maximum gap length for training
    #target with forecast horizon not needed ATM
    features_train, labels_train = timeseries_dataset_from_array(X_train_sc, forecast_horizon, horizon, AR=True)
    features_val, labels_val=timeseries_dataset_from_array(X_val_sc, forecast_horizon, horizon, AR=True) 
    # labels_train = features_train
    # labels_val = features_val
    
    #get data_loader for all data, data_loader is an torch iterable to be able to iterate over batches
    dataset_train, data_loader_train = get_dataloader(features_train, labels_train, batch_size=batch_size)
    dataset_val, data_loader_val=get_dataloader(features_val, labels_val, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???
    
    
    #############################################################
    #set up an autregressive model - the autoregressive model loop is defined in AR_model.py. The class in there imports a neural network configuration that is defined inside AR_nn_models.py, a feed forward model with 2 layers and as many neurons as defined in hidden size
    input_size=features_train.shape[-1]  #number of input features 
    hidden_size=25 #number of neurons in hidden layer
    output_size=horizon 
    
    model=sampleFFNN_AR(input_size, hidden_size, output_size) 
    tr=True
    ###################
    if tr==True:
        trainer = Trainer(model,epochs,respath, batch_size)
        trainer.fit(data_loader_train,data_loader_val)
    ###################

    #plot losses
    plot_losses(losslogpath, respath)
    
    # ###################################################################
    # ###################### Testing
    
    
    #load weights of best model
    model.load_state_dict(torch.load(os.path.join(respath,'weights.pth')))
    model.eval()
    #create test features and label
    features_test, labels_test =timeseries_dataset_from_array(X_test_sc, forecast_horizon, horizon, AR=True)
    dataset_test, data_loader_test=get_dataloader(features_test, labels_test, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???
    
    #generate predictions based on test set
    all_test_preds=[]
    for step, (x_batch_test, y_batch_test) in enumerate(data_loader_test):
        #generate prediction
        pred=model(x_batch_test)
        #calculate metric for each batch, exlcude precipitation in target
        all_test_preds.append(pred)

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

    plt.figure()   
    plt.plot(pd.to_datetime(X_test.index), plot_test, label='prediction')
    plt.plot(pd.to_datetime(X_test.index), X_test[test_id], label='observation')
    plt.xlabel('Date')
    plt.ylabel('Water level [m]')
    plt.legend()  
    plt.title(f'Model: {model_name}')
    plt.show()
    plt.savefig(os.path.join(respath, 'preds_test.png'))
    # plt.close()    



    #plot whole prediction horizon of a timestep (=time of prediciton TOP)
    date=pd.to_datetime(X_test.index)
    for i in np.arange(0,len(preds_test_unsc), 500):
        plt.figure()
        #add nan before TOP
        plot_TOP_preds=np.concatenate(([np.nan],  preds_test_unsc[i]))
        X_TOP_unsc=train_WL_sc.inverse_transform(X_test_sc[i:i+forecast_horizon+1,0].reshape(-1,1))
        plt.plot(date[i:i+forecast_horizon+1],plot_TOP_preds, label='prediction', linestyle='None', marker='.')
        plt.plot(date[i:i+forecast_horizon+1],X_TOP_unsc[:,0], label='observation', linestyle='None', marker='.')
        #Plot vertical line highlighting TOP
        plt.axvline(x=date[i], color='black', linestyle='--', label='TOP')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Water level [m]')
        plt.title(f'Time step {i}: {date[i]}')
        plt.show()
        plt.savefig(os.path.join(respath, f'preds_TOP_{i}.png'))
        # plt.close()
         
        
    


#extract first time step of each data window
plot_preds_test=train_WL_sc.inverse_transform(preds_test[:,0,:])
plot_labels_test=train_WL_sc.inverse_transform(labels_test[:,0,:1])


plt.figure()   
plt.plot(pd.to_datetime(X_test.index)[1:-forecast_horizon+1], plot_preds_test, label='prediction')
plt.plot(pd.to_datetime(X_test.index)[1:-forecast_horizon+1], plot_labels_test, label='observation')
plt.xlabel('Date')
plt.ylabel('Water level [m]')
plt.legend()  
plt.title(f'Model: {model_name}')
plt.show()

fig, ax1 = plt.subplots()

# Plot water level on the bottom axis
ax1.plot(pd.to_datetime(X_test.index), X_test[test_id], color='blue', label='Observation')
ax1.plot(pd.to_datetime(X_test.index), plot_test, color='orange', label='Prediction')
ax1.set_xlabel('Date')
ax1.set_ylabel('Water Level')
ax1.tick_params('y')

# Create a second y-axis for precipitation
ax2 = ax1.twinx()
ax2.plot(-X_test[test_prcp], color='green', label='Precipitation')
ax2.set_ylabel('Precipitation')
ax2.tick_params('y')
ax2.set_ylim(-20, 0)

# Invert the tick labels on the second y-axis such that they are displayed positive
yticks = ax2.get_yticks()
ax2.set_yticks(yticks)
ax2.set_yticklabels([abs(y) for y in yticks])
fig.legend()





##########################Gap study#################
# gap=False

# if gap:   
#     # create gaps in features (only WL not rain)
#     features_test_gaps=generate_gaps_features(features_test.copy(), 0.1)
#     #create dataloader with gaps in features, but not in labels
#     dataset_test, data_loader_test=get_dataloader(features_test_gaps, features_test[:,:,:1], batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???

#     #toDo fix scaling
#     plt.figure()
#     for step, (inputs,labels) in enumerate(data_loader_test):
#         print(np.isnan(inputs).sum())
#         plt.plot(labels[0,:,0], color='cyan',label='observation')
#         plt.plot(inputs[0,:,0], color='blue' ,label='observations with artifical gaps')
#         preds = model(inputs).detach().numpy()
#         preds=np.concatenate((np.full((1,1,1),np.nan), preds), axis=1)
#         plt.plot(preds[0,:,0], color='red' ,label='prediction')

#     plt.legend()
#     plt.xlabel('Date')
#     plt.ylabel('Water level (scaled)')

# else:
#     dataset_test, data_loader_test=get_dataloader(features_test, labels_test, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???
#     #initialise arrays for storing predictions
#     test_preds1=np.zeros(0)
#     test_preds2=np.zeros(0)
#     for step, (inputs,labels) in enumerate(data_loader_test):
#         preds = model(inputs, labels).detach().numpy()
#         # unscale and save data, first prediction based on observations
#         test_preds1=np.append(test_preds1, rescale_data(preds[:,:1,:], train_sc, 2)[:,0])
#         #unscale and save data, predictions with one predicted timestep fed back
#         test_preds2=np.append(test_preds2, rescale_data(preds[:,1:,:], train_sc, 2)[:,0])
