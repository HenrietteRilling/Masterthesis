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
from utils import scale_data, timeseries_dataset_from_array, get_dataloader
from plot_utils import plot_losses
from AR_model import sampleFFNN_AR, sampleLSTM_AR, sampleFFNN_AR_2
from AR_trainer import Trainer, Trainer_AR_2

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

#####################################################
#introduce artificial gaps in WL test 
# X_test_sc=generate_gaps(X_test_sc, 0.1)

#############################
#choose which model is tested, options: FFNN_AR, FFNN_AR1, FFNN_AR2
model_tested='FFNN_AR1'

# =============================================================================
# FFNN_AR: autoregressive model, predicting one timestep ahead
# =============================================================================

if model_tested=='FFNN_AR':

    respath=f'./results/results_{model_tested}'
    if not os.path.exists(respath): os.makedirs(respath)
    #delete any existing losslog/files, to only save losses of current model run
    losslogpath=os.path.join(respath, 'losslog.csv')
    if os.path.exists(losslogpath): os.remove(losslogpath)
    
    windowsize=24
    horizon=1 #how many timesteps in the future do we want to predict
    max_gap_length=0
    epochs=10
    batch_size=100 #number of batches that is processed at once 
    
    #batch data
    #get input in batches with w timesteps 
    #we feed predictions back in case of gaps, therefore we need a to extend the target by the maximum gap length for training
    #target with forecast horizon not needed ATM
    features_train, labels_train = timeseries_dataset_from_array(X_train_sc, windowsize, horizon+max_gap_length, AR=True)
    features_val, labels_val=timeseries_dataset_from_array(X_val_sc, windowsize, horizon+max_gap_length, AR=True) 
    labels_train = features_train
    labels_val = features_val
    
    #get data_loader for all data, data_loader is an torch iterable to be able to iterate over batches
    dataset_train, data_loader_train = get_dataloader(features_train, labels_train, batch_size=batch_size)
    dataset_val, data_loader_val=get_dataloader(features_val, labels_val, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???
    
    
    #############################################################
    #set up an autregressive model - the autoregressive model loop is defined in AR_model.py. The class in there imports a neural network configuration that is defined inside AR_nn_models.py, a feed forward model with 2 layers and as many neurons as defined in hidden size
    input_size=features_train.shape[-1]+1  #number of input features, nr of neurons (nr of features +1 as model is AR: predictions are added as additional feature in forward step)
    hidden_size=25
    output_size=1
    
    model=sampleFFNN_AR(input_size, hidden_size, output_size) 
    tr=True
    ###################
    if tr==True:
        trainer = Trainer(model,epochs,respath)
        trainer.fit(data_loader_train,data_loader_val)
    ###################
    
    ###################################################################
    ###################### Testing
    
    
    #load weights of best model
    model.load_state_dict(torch.load(os.path.join(respath,'weights.pth')))
    model.eval()
    #create test features and label
    features_test, labels_test =timeseries_dataset_from_array(X_test_sc, len(X_test_sc)-1, horizon+max_gap_length, AR=True) 

    #generate predictions based on test set
    preds_test=model(torch.tensor(features_test).float(), torch.tensor(labels_test).float())

    preds_test_unsc=train_WL_sc.inverse_transform(preds_test[0,:,:].detach().numpy())
    plot_preds_test_unsc=np.concatenate((np.full(horizon, np.nan),preds_test_unsc[:,0]))
    plt.figure()
    plt.plot(pd.to_datetime(X_test.index), plot_preds_test_unsc, label='prediction')
    plt.plot(pd.to_datetime(X_test.index), X_test[test_id], label='observation')
    plt.xlabel('Date')
    plt.ylabel('Water level [m]')
    plt.legend()
    
    
    ##if target=labels:
    # preds_test_unsc=train_WL_sc.inverse_transform(preds_test[0,:,:].detach().numpy())
    # plot_preds_test_unsc=np.concatenate((preds_test_unsc[:,0], np.full(horizon, np.nan)))
    # plt.figure()
    # plt.plot(pd.to_datetime(X_test.index), plot_preds_test_unsc, label='prediction')
    # plt.plot(pd.to_datetime(X_test.index), X_test[test_id], label='observation')
    # plt.xlabel('Date')
    # plt.ylabel('Water level [m]')
    # plt.legend()   
    
# =============================================================================
# FFNN_AR1: AR model, predicting h timesteps ahead in the future
# =============================================================================    
    
elif model_tested=='FFNN_AR1':

     respath=f'./results/results_{model_tested}'
     if not os.path.exists(respath): os.makedirs(respath)
     #delete any existing losslog/files, to only save losses of current model run
     losslogpath=os.path.join(respath, 'losslog.csv')
     if os.path.exists(losslogpath): os.remove(losslogpath)
     
     windowsize=48
     horizon=12 #how many timesteps in the future do we want to predict
     max_gap_length=0
     epochs=20
     batch_size=100 #number of batches that is processed at once 
     
     #batch data
     #get input in batches with w timesteps 
     #we feed predictions back in case of gaps, therefore we need a to extend the target by the maximum gap length for training
     #target with forecast horizon not needed ATM
     features_train, labels_train = timeseries_dataset_from_array(X_train_sc, windowsize, horizon+max_gap_length, AR=True)
     features_val, labels_val=timeseries_dataset_from_array(X_val_sc, windowsize, horizon+max_gap_length, AR=True) 
     # labels_train = features_train
     # labels_val = features_val
     
     #get data_loader for all data, data_loader is an torch iterable to be able to iterate over batches
     dataset_train, data_loader_train = get_dataloader(features_train, labels_train, batch_size=batch_size)
     dataset_val, data_loader_val=get_dataloader(features_val, labels_val, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???
     
     
     #############################################################
     #set up an autregressive model - the autoregressive model loop is defined in AR_model.py. The class in there imports a neural network configuration that is defined inside AR_nn_models.py, a feed forward model with 2 layers and as many neurons as defined in hidden size
     input_size=features_train.shape[-1]+horizon  #number of input features, nr of neurons (nr of features +horizon as model is AR: predictions are added as additional feature in forward step)
     hidden_size=25
     output_size=horizon
     
     model=sampleFFNN_AR(input_size, hidden_size, horizon) 
     tr=True
     ###################
     if tr==True:
         trainer = Trainer(model,epochs,respath)
         trainer.fit(data_loader_train,data_loader_val)
     ###################
     
     ###################################################################
     ###################### Testing
     
     #load weights of best model
     model.load_state_dict(torch.load(os.path.join(respath,'weights.pth')))
     model.eval()
     #create test features and label
     features_test, labels_test =timeseries_dataset_from_array(X_test_sc, len(X_test_sc)-horizon, horizon+max_gap_length, AR=True) 
     
     #generate predictions based on test set
     preds_test=model(torch.tensor(features_test).float(), torch.tensor(labels_test).float())
     
     preds_test_unsc=train_WL_sc.inverse_transform(preds_test[0,:,:].detach().numpy())
     plot_preds_test_unsc=np.concatenate((np.full(horizon, np.nan),preds_test_unsc[:,0]))
     plt.figure()
     plt.plot(pd.to_datetime(X_test.index), plot_preds_test_unsc, label='prediction')
     plt.plot(pd.to_datetime(X_test.index), X_test[test_id], label='observation')
     plt.xlabel('Date')
     plt.ylabel('Water level [m]')
     plt.legend()
       


# =============================================================================
# FFNN_AR2: AR model, predictions are transferred between Batches
# =============================================================================


elif model_tested=='FFNN_AR2':

    respath=f'./results/results_{model_tested}'
    if not os.path.exists(respath): os.makedirs(respath)
    #delete any existing losslog/files, to only save losses of current model run
    losslogpath=os.path.join(respath, 'losslog.csv')
    if os.path.exists(losslogpath): os.remove(losslogpath)

    windowsize=24 #one day
    horizon=10 #how many timesteps in the future do we want to predict
    max_gap_length=0
    epochs=3
    batch_size=100 #number of batches that is processed at once 

    ##########################################
    #batch data
    #get input in batches with w timesteps 
    #we feed predictions back in case of gaps, therefore we need a to extend the target by the maximum gap length for training
    features_train, labels_train = timeseries_dataset_from_array(X_train_sc, windowsize, horizon+max_gap_length)
    features_val, labels_val=timeseries_dataset_from_array(X_val_sc, windowsize, horizon+max_gap_length) 
    
    #get data_loader for all data, data_loader is an torch iterable to be able to iterate over batches
    dataset_train, data_loader_train = get_dataloader(features_train, labels_train, batch_size=batch_size)
    dataset_val, data_loader_val=get_dataloader(features_val, labels_val, batch_size=batch_size)
    
    #############################################################
    #model initialization, FFNN with 2 layers, and as many neurons as defined in hidden_size
    input_size=features_train.shape[-1]+1  #number of input features, (nr of features +1 as model is AR: predictions are added as additional feature in forward step)
    hidden_size=25 #nr of neurons
    
    model=sampleFFNN_AR_2(input_size, hidden_size, horizon) 
    tr=True
    ###################
    if tr==True:
        trainer = Trainer_AR_2(model,epochs,respath, batch_size)
        trainer.fit(data_loader_train,data_loader_val)
    ###################
    
    ##plot losses
    plot_losses(losslogpath)

    ###################################################################
    ###################### Testing
    
    
    #load weights of best model
    model.load_state_dict(torch.load(os.path.join(respath,'weights.pth')))
    model.eval()
    #create test features and label
    features_test, labels_test =timeseries_dataset_from_array(X_test_sc, windowsize, horizon+max_gap_length)
    dataset_test, data_loader_test=get_dataloader(features_test, labels_test, batch_size=batch_size, shuffle=False)


    all_preds=[]
    y_hat=torch.zeros(batch_size,horizon,1)
    for step, (x_batch_test, y_batch_test) in enumerate(data_loader_test):
        if step==len(data_loader_test)-1 and x_batch_test.size(0)!=batch_size:
            break
        preds_test=model(x_batch_test, y_hat)
        all_preds.append(preds_test)
        y_hat=preds_test

    #extract first prediciton of the h time steps we predict ahead
    all_preds_newest=[torch.squeeze(p[:,:1,:]) for p in all_preds]
    #concat list among timeaxis to have one tensor
    all_preds_newest=torch.cat(all_preds_newest,0)
    
    #unscale data
    all_preds_newest_unsc=train_WL_sc.inverse_transform(all_preds_newest.detach().numpy().reshape(-1,1))
    #add nans in length of windowsize, and at the end for last batch that was excluded
    plot_all_test=np.concatenate((np.full(windowsize, np.nan),all_preds_newest_unsc[:,0], np.full(len(X_test_sc)-windowsize-len(all_preds_newest), np.nan)))

    plt.figure()
    plt.plot(pd.to_datetime(X_test.index), plot_all_test, label='prediction')
    plt.plot(pd.to_datetime(X_test.index), X_test[test_id], label='observation')
    plt.xlabel('Date')
    plt.ylabel('Water level [m]')
    plt.legend()        
    
    
else:
    print('No valid model chosen, check variable model_tested')

###################################################################
###################### Testing

#load weights of best model
model.load_state_dict(torch.load(os.path.join(respath,'weights.pth')))
model.eval()
#create test features and label
features_test, labels_test =timeseries_dataset_from_array(X_test_sc, len(X_test_sc)-horizon, horizon+max_gap_length, AR=True) 

#generate predictions based on test set
preds_test=model(torch.tensor(features_test).float(), torch.tensor(labels_test).float())

preds_test_unsc=train_WL_sc.inverse_transform(preds_test[0,:,:].detach().numpy())
plot_preds_test_unsc=np.concatenate((preds_test_unsc[:,0], np.full(horizon, np.nan)))
plt.figure()
plt.plot(pd.to_datetime(X_test.index), plot_preds_test_unsc, label='prediction')
plt.plot(pd.to_datetime(X_test.index), X_test[test_id], label='observation')
plt.xlabel('Date')
plt.ylabel('Water level [m]')
plt.legend()

date=pd.to_datetime(X_test.index)
for i in np.arange(0,100, 10):
    plt.figure()
    #add nans before TOP
    # plot_TOP_preds=np.concatenate((np.full(windowsize, np.nan),  preds_test_unsc[i]))
    X_TOP_unsc=train_WL_sc.inverse_transform(X_test_sc[i:i+horizon,0].reshape(-1,1))
    # plot_TOP_obs=np.concatenate((X_test_sc[:windowsize,0], np.full(horizon, np.nan)))
    plt.plot(date[i:i+horizon],preds_test_unsc[i], label='prediction', linestyle='None', marker='.')
    plt.plot(date[i:i+horizon],X_TOP_unsc[:,0], label='observation', linestyle='None', marker='.')
    #Plot vertical line highlighting TOP
    plt.axvline(x=date[i], color='black', linestyle='--', label='TOP')
    plt.legend()





# #remove last dimension from the predictions such that consecutive time steps are in one array
# preds_TOP_unsc=np.squeeze(preds_test.detach().numpy(), axis=2)

# #plot predictions starting from last observation (=Time of prediction, TOP)
# for i in np.arange(0,1001, 1000):
#     plt.figure()
#     #add nans before TOP
#     plot_TOP_preds=np.concatenate((np.full(windowsize, np.nan),  preds_TOP_unsc[i]))
#     # plot_TOP_obs=np.concatenate((X_test_sc[:windowsize,0], np.full(horizon, np.nan)))
#     plt.plot(plot_TOP_preds, label='prediction', linestyle='None', marker='.')
#     plt.plot(X_test_sc[i:i+windowsize+horizon,0], label='observation', linestyle='None', marker='.')
#     #Plot vertical line highlighting TOP
#     plt.axvline(x=windowsize-1, color='black', linestyle='--', label='TOP')
#     plt.legend()



# #############################################################
# #set up an autregressive LSTM model - the autoregressive model loop is defined in AR_model.py. The class in there imports a neural network configuration that is defined inside AR_nn_models.py, this is a feed forward model with 2 layers of 25 neurons
# respath2='./results_LSTM'
# if not os.path.exists(respath): os.makedirs(respath)

# #delete any existing losslog/files, to only save losses of current model run
# losslogpath=os.path.join(respath2, 'losslog.csv')
# if os.path.exists(losslogpath): os.remove(losslogpath)

# input_size=features_train.shape[-1]+1
# hidden_size=2 
# num_layers=1
# model=sampleLSTMmodel(input_size, hidden_size, num_layers) #number of input features, nr of neurons (nr of features +1 as model is AR: predictions are added as additional feature in forward step)

# tr=True
# ###################
# if tr==True:
#     trainer = Trainer(model,epochs,respath2)
#     trainer.fit(data_loader_train,data_loader_val)
# ###################


# ###################################################################
# ###################### Testing


# #load weights of best model
# model.load_state_dict(torch.load(os.path.join(respath2,'weights.pth')))
# #create test features and label
# features_test, labels_test=timeseries_dataset_from_array(X_test_sc, windowsize, horizon+max_gap_length) 

# #generate predictions based on test set
# preds_test=model(torch.tensor(features_test).float(), torch.tensor(labels_test).float())

# #remove last dimension from the predictions such that consecutive time steps are in one array
# preds_TOP_unsc=np.squeeze(preds_test.detach().numpy(), axis=2)

# #plot predictions starting from last observation (=Time of prediction, TOP)
# for i in np.arange(0,1001, 100):
#     plt.figure()
#     #add nans before TOP
#     plot_TOP_preds=np.concatenate((np.full(windowsize, np.nan),  preds_TOP_unsc[i]))
#     # plot_TOP_obs=np.concatenate((X_test_sc[:windowsize,0], np.full(horizon, np.nan)))
#     plt.plot(plot_TOP_preds, label='prediction', linestyle='None', marker='.')
#     plt.plot(X_test_sc[i:i+windowsize+horizon,0], label='observation', linestyle='None', marker='.')
#     #Plot vertical line highlighting TOP
#     plt.axvline(x=windowsize-1, color='black', linestyle='--', label='TOP')
#     plt.legend()



# plt.figure()
# plt.plot(X_test_sc[:,0], label='observation')
# plt.plot(preds_test[:,0,0].detach().numpy(), label='prediction')
# plt.legend()

# preds_test_unscaled=preds_test.detach().numpy()
# plcholder1=np.zeros(windowsize)*np.nan #for first prediction based on observations
# plcholder2=np.zeros(windowsize+1)*np.nan #for second prediction
# plcholder3=np.zeros(windowsize+2)*np.nan #for third prediction (current example h=3)

# test_preds1_plot=np.concatenate((plcholder1,preds_test_unscaled[:,0,0], np.full(2, np.nan)))
# test_preds2_plot=np.concatenate((plcholder2,preds_test_unscaled[:,1,0], np.full(1, np.nan)))
# test_preds3_plot=np.concatenate((plcholder3,preds_test_unscaled[:,2,0]))

# plt.figure()
# plt.plot(X_test_sc[:,0], label='observation')
# plt.plot(test_preds1_plot, label='prediction 1')
# plt.plot(test_preds2_plot, label='prediction 2')
# plt.plot(test_preds3_plot, label='prediction 3')
# plt.legend()





#test=np.concatenate((plcholder1, preds_test_unscaled))
# #first prediction is made after windowsize
# plcholder1=np.zeros(windowsize+horizon)*np.nan
# plcholder2=np.zeros(windowsize+horizon+1)*np.nan
# test_preds1_plot=np.concatenate((plcholder1, test_preds1))
# test_preds2_plot=np.concatenate((plcholder2, test_preds2[:-1]))
# X_test['pred1']=test_preds1_plot
# X_test['pred2']=test_preds2_plot
# plt.figure()
# plt.plot(X_test[test_id],label='observation')
# plt.plot(X_test['pred1'], color='red', label='prediction')
# plt.plot(X_test['pred2'], color='green', label='prediction2')
# plt.legend()
# plt.xlabel('Date')
# plt.ylabel('Water level [m]')

# plcholder1=np.zeros(windowsize)*np.nan
# plcholder2=np.zeros(windowsize)*np.nan
# plcholder3=np.full(horizon, np.nan)
# test_preds1_plot=np.concatenate((plcholder1, test_preds1, plcholder3))
# test_preds2_plot=np.concatenate((plcholder2, test_preds2, plcholder3))
# X_test['pred1']=test_preds1_plot
# X_test['pred2']=test_preds2_plot
# plt.figure()
# plt.plot(X_test[test_id],label='observation')
# plt.plot(X_test['pred1'], color='red', label='prediction')
# plt.plot(X_test['pred2'], color='green', label='prediction2')
# plt.legend()
# plt.xlabel('Date')
# plt.ylabel('Water level [m]')
    

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
