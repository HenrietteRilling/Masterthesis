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

from Data_loader import get_WL_data, get_prcp_data, get_test_data
from utils import scale_data, rescale_data, rescale_data_ffn, timeseries_dataset_from_array, get_dataloader
from plot_utils import datetime_xaxis
from AR_model import samplemodel
from AR_trainer import Trainer

def generate_gaps(data, number):
    size=int(number*len(data))
    gapidx=np.random.randint(len(data),size=size)
    data[gapidx,0]=np.nan
    return data  
    
def generate_gaps_features(data, number):
    #number of gaps relative to the length of the data
    no_of_gaps=int(number*len(data[0,:,0]))
    #choose randomly indexes where a gap will be created, np.choice is used as np.random.randint samples with replacement
    gapidx=np.random.choice(np.arange(0, len(data[0,:,0])),no_of_gaps,replace=False)  
    #replace values at chosen timesteps with nans
    data[:,gapidx,0]=np.nan
    return data

windowsize=10
horizon=1 #how many timesteps in the future do we want to predict
max_gap_length=1
epochs=10
batch_size=100 #number of batches that is processed at once 
#
respath='./results'
if not os.path.exists(respath): os.makedirs(respath)

#delete any existing losslog/files, to only save losses of current model run
losslogpath=os.path.join(respath, 'losslog.csv')
if os.path.exists(losslogpath): os.remove(losslogpath)

#############################################
#load data
WL, _, station_name_to_id, _ = get_WL_data(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\Data_WL')
prcp=get_prcp_data(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\SVK', r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\DMI_Climate_Data_prcp', join=True)

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

#scale and normalise such that all data has a range between [0,1], store scaler for rescaling
X_train_sc, train_sc = scale_data(X_train)
X_val_sc = train_sc.transform(X_val)
X_test_sc = train_sc.transform(X_test)

#####################################################
#introduce artificial gaps in WL data

# X_train_sc=generate_gaps(X_train_sc, 0.1)
# X_val_sc=generate_gaps(X_val_sc, 0.1)
# X_test_sc=generate_gaps(X_test_sc, 0.1)

##########################################
#batch data
#get input and targets in batches with 10 timesteps input and predict the next timestep t+1, prcp data is needed in the target for feeding back predictions
#we feed predictions back in case of gaps, therefore we need a to extend the target by the maximum gap length for training
features_train, labels_train = timeseries_dataset_from_array(X_train_sc, windowsize, horizon+max_gap_length)
features_val, labels_val=timeseries_dataset_from_array(X_val_sc, windowsize, horizon+max_gap_length) 

#get data_loader for all data, data_loader is an torch iterable to be able to iterate over batches
# dataset_train, data_loader_train = get_dataloader(features_train, labels_train, batch_size=batch_size)
# dataset_val, data_loader_val=get_dataloader(features_val, labels_val, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???

#features and lables have to be identical (same periods of time) in order to adopt Roland's code, however, labels should contain only water level data

dataset_train, data_loader_train = get_dataloader(features_train, labels_train, batch_size=batch_size)
dataset_val, data_loader_val=get_dataloader(features_val, labels_val, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???


#############################################################
#set up an autregressive model - the autoregressive model loop is defined in AR_model.py. The class in there imports a neural network configuration that is defined inside AR_nn_models.py, this is a feed forward model with 2 layers of 25 neurons
model=samplemodel(features_train.shape[-1], 25) #4 = number of inputs in the linear layer, 3 as we input 2 timesteps a 2 features
tr=True
###################
if tr==True:
    trainer = Trainer(model,epochs,respath)
    trainer.fit(data_loader_train,data_loader_val)
###################


#load weights of best model
model.load_state_dict(torch.load(os.path.join(respath,'weights.pth')))
#create test features and label
features_test, labels_test=timeseries_dataset_from_array(X_test_sc, windowsize, horizon+max_gap_length) 

gap=False

if gap:   
    # create gaps in features (only WL not rain)
    features_test_gaps=generate_gaps_features(features_test.copy(), 0.1)
    #create dataloader with gaps in features, but not in labels
    dataset_test, data_loader_test=get_dataloader(features_test_gaps, features_test[:,:,:1], batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???

    #toDo fix scaling
    plt.figure()
    for step, (inputs,labels) in enumerate(data_loader_test):
        print(np.isnan(inputs).sum())
        plt.plot(labels[0,:,0], color='cyan',label='observation')
        plt.plot(inputs[0,:,0], color='blue' ,label='observations with artifical gaps')
        preds = model(inputs).detach().numpy()
        preds=np.concatenate((np.full((1,1,1),np.nan), preds), axis=1)
        plt.plot(preds[0,:,0], color='red' ,label='prediction')

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Water level (scaled)')

else:
    dataset_test, data_loader_test=get_dataloader(features_test, labels_test, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???
    #initialise arrays for storing predictions
    test_preds1=np.zeros(0)
    test_preds2=np.zeros(0)
    for step, (inputs,labels) in enumerate(data_loader_test):
        preds = model(inputs, labels).detach().numpy()
        # unscale and save data, first prediction based on observations
        test_preds1=np.append(test_preds1, rescale_data(preds[:,:1,:], train_sc, 2)[:,0])
        #unscale and save data, predictions with one predicted timestep fed back
        test_preds2=np.append(test_preds2, rescale_data(preds[:,1:,:], train_sc, 2)[:,0])

#first prediction is made after windowsize
plcholder1=np.zeros(windowsize+horizon)*np.nan
plcholder2=np.zeros(windowsize+horizon+1)*np.nan
test_preds1_plot=np.concatenate((plcholder1, test_preds1))
test_preds2_plot=np.concatenate((plcholder2, test_preds2[:-1]))
X_test['pred1']=test_preds1_plot
X_test['pred2']=test_preds2_plot
plt.figure()
plt.plot(X_test[test_id],label='observation')
plt.plot(X_test['pred1'], color='red', label='prediction')
plt.plot(X_test['pred2'], color='green', label='prediction2')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Water level [m]')

plcholder1=np.zeros(windowsize-1)*np.nan
plcholder2=np.zeros(windowsize-1)*np.nan
test_preds1_plot=np.concatenate((plcholder1, test_preds1, [np.nan, np.nan]))
test_preds2_plot=np.concatenate((plcholder2, test_preds2, [np.nan, np.nan]))
X_test['pred1']=test_preds1_plot
X_test['pred2']=test_preds2_plot
plt.figure()
plt.plot(X_test[test_id],label='observation')
plt.plot(X_test['pred1'], color='red', label='prediction')
plt.plot(X_test['pred2'], color='green', label='prediction2')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Water level [m]')
    
