# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:09:19 2023

@author: Henriette
"""

import os, sys
import numpy as np
import datetime, time
import torch
import matplotlib.pyplot as plt
#Set the number of CPU threads that PyTorch will use for parallel processing
torch.set_num_threads(8)

from Data_loader import get_WL_data, get_prcp_data, get_test_data
from utils import scale_data, rescale_data, timeseries_dataset_from_array, get_dataloader
from AR_model import samplemodel
from AR_trainer import Trainer

windowsize=25
horizon=1 #how many timesteps in the future do we want to predict
epochs=10
batch_size=100 #number of batches that is processes at once

#
respath='./results'
if not os.path.exists(respath): os.makedirs(respath)

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
test_station='ns Uldumkær'
test_id=station_name_to_id.get(test_station)
test_prcp='05225'

X_WL=get_test_data(test_id, WL_wo_anom)
X_prcp=get_test_data(test_prcp, prcp)

#merge precipitation and WL data, select overlapping timeperiod
# X=pd.concat([X_WL, X_prcp], axis=1).loc[X_WL.index.intersection(X_prcp.index)]
X=X_WL 

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

##########################################
#batch data
#get input and targets in batches with 10 timesteps input and predict the next timestep t+1, prcp data is only important for input, therefore label
features_train, labels_train = timeseries_dataset_from_array(X_train_sc, windowsize, horizon, label_indices=[0])
features_val, labels_val=timeseries_dataset_from_array(X_val_sc, windowsize, horizon, label_indices=[0]) 

#get data_loader for all data, data_loader is an torch iterable to be able to iterate over batches
# dataset_train, data_loader_train = get_dataloader(features_train, labels_train, batch_size=batch_size)
# dataset_val, data_loader_val=get_dataloader(features_val, labels_val, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???

#features and lables have to be identical in order to adopt Roland's code
dataset_train, data_loader_train = get_dataloader(features_train, features_train, batch_size=batch_size)
dataset_val, data_loader_val=get_dataloader(features_val, features_val, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???


#############################################################
#set up an autregressive model - the autoregressive model loop is defined in AR_model.py. The class in there imports a neural network configuration that is defined inside AR_nn_models.py, this is a feed forward model with 2 layers of 25 neurons
model=samplemodel(2, 25)

###################
if True:
    trainer = Trainer(model,epochs,respath)
    trainer.fit(data_loader_train,data_loader_val)
###################

model.load_state_dict(torch.load(os.path.join(respath,'weights.pth')))

features_test, labels_test=timeseries_dataset_from_array(X_test_sc, len(X_test_sc)-horizon, horizon, label_indices=[0]) 
dataset_test, data_loader_test=get_dataloader(features_test, features_test, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???

plt.figure()
for step, (inputs,labels) in enumerate(data_loader_test):
    preds = model(inputs,labels).detach().numpy()
    # import pdb
    # pdb.set_trace()
    # unscale data
    preds=train_sc.inverse_transform(preds[0,:,:])
    labels=train_sc.inverse_transform(labels[0,:,:].numpy())
    plt.plot(labels,label='observation')
    plt.plot(preds,color='red', label='prediction')
plt.legend()


