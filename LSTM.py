# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:08:08 2023

@author: Henriette
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from Data_loader import get_WL_data, get_prcp_data
from window_data import timeseries_dataset_from_array, _get_labelled_window


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_layers = num_layers #number of stacked LSTM layers
        self.input_size = input_size #number of expected features in the input x
        self.hidden_size = hidden_size #number of features in the hidden state h

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #Defintion of the LSTM
        self.fc =  nn.Linear(hidden_size, 1) #fully connected last layer, combines input to one output
     
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm1(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.fc(hn) #Final output
        return out.unsqueeze(-1) #unsqueeze adds another dimension of 1 to the tensor, necessary to have same shape as batched target data

def get_test_data(stat_id, data_df):
    '''get data for specific station (e.g. WL, precipitation), cropped to their actual timeframe
    data_df: dataframe, timeseries data with  station_ids as column names'''
    
    X=data_df[[stat_id]]
    
    #make sure that only period in which sensor data is available is used
    X=X[(X.index>X.first_valid_index())&(X.index<X.last_valid_index())]
    #interpolate missing values
    X.interpolate(inplace=True)
    return X

def scale_data(data):
    scaler=MinMaxScaler()
    scaled_data=scaler.fit_transform(data)
    return scaled_data, scaler

def rescale_data(data, scaler):
    rescaled_data=scaler.inverse_transform(data)
    return rescaled_data

def plot_predictions(epoch, y, y_pred_train, y_pred_val, train_start, train_end):
    plt.figure()
    plt.plot(y, label='observation', color='b')
    
    #Training data
    train_plot=np.ones_like(y) * np.nan
    #add another dimension
    train_plot=train_plot[:,None]
    train_plot[train_start:train_end]=y_pred_train[:,-1,:]
    plt.plot(train_plot, label='train')   
    
    #validation data
    val_plot=np.ones_like(y) * np.nan
    #add another dimension
    val_plot=val_plot[:,None]
    val_plot[train_end+train_start:len(y)]=y_pred_val[:,-1,:]
    plt.plot(val_plot, label='validation')
    plt.title(f'Epoch: {epoch}')
    plt.legend()
    plt.show()    
    
    
'''Data Preprocessing'''
#Load data
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
X=pd.concat([X_WL, X_prcp], axis=1).loc[X_WL.index.intersection(X_prcp.index)]

#split in train, test and val data
#note: pd slicing is inlcusive
X_train=X['2011-01-01':'2018-12-31']
X_val=X['2019-01-01':'2021-12-31']
X_test=X['2022-01-01':'2022-12-31']
#combine train and val data for plotting
X_test_all=X['2011-01-01':'2021-12-31']
X_test_all=X_test_all[test_id].to_numpy()

#scale and normalise such that all data has a range between [0,1], store scaler for rescaling
X_train_sc, train_sc = scale_data(X_train)
X_val_sc, val_sc = scale_data(X_val)
X_test_sc, test_sc = scale_data(X_test)
#concat scaled timeseries for plotting
X_sc_complete=np.append(X_train_sc[:,0], X_val_sc[:,0])

'''Batching data'''
#parameter defintion
window_size=10
horizon=1

#get input and targets in batches with 10 timesteps input and predict the next timestep t+1, prcp data is only important for input, therefore label
features_train, labels_train = timeseries_dataset_from_array(X_train_sc, window_size, horizon, label_indices=[0])
features_val, labels_val=timeseries_dataset_from_array(X_val_sc, window_size, horizon, label_indices=[0]) 

dataset = torch.utils.data.TensorDataset(torch.tensor(features_train), torch.tensor(labels_train)) # insert into tensor dataset
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True) # insert dataset into data loader
#batch_size (int, optional) – how many samples per batch to load (default: 1)

'''LSTM training + SetUp'''
#defintion of hyperparameters
num_epochs = 100 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 2 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

#Instantiate  the class LSTM object.
model = LSTM(input_size, hidden_size, num_layers) #our lstm class

#Choose loss function and Optimizer
loss_fn = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

#initialize variable to keep track of the best model
best_val_loss=100
#lists to store losses
train_losses=[]
val_losses=[]


#Training of the model
for epoch in range(num_epochs):
    counter = 0 
    model.train()
    for X_batch, y_batch in data_loader:
        counter+=1
        if torch.any(X_batch.isnan()):
            print(f'X_batch is nan: {X_batch}')
        X_batch=X_batch.float()
        y_batch=y_batch.float()
        y_pred=model(X_batch)
        optimizer.zero_grad()
        loss=loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    model.eval()
    with torch.no_grad():
        '''Train data'''
        y_pred_train = model(torch.tensor(features_train).float())
        train_rsme=np.sqrt(loss_fn(y_pred_train, torch.tensor(labels_train).float()))
        train_losses.append(train_rsme)

        #rescale data to actual range
        #y_pred = rescale_data(y_pred, train_sc) #To Do
       
        '''Validation data'''
        y_pred_val = model(torch.tensor(features_val).float())
        val_rsme=np.sqrt(loss_fn(y_pred_val, torch.tensor(labels_val).float()))
        val_losses.append(val_rsme)
        if val_rsme < best_val_loss:
            best_val_loss=val_rsme
            best_model=model.state_dict()

        #rescale data to actual range
        # y_pred=rescale_data(y_pred, val_sc) #To Do
        if epoch % 10 ==0:
            plot_predictions(epoch, X_sc_complete, y_pred_train, y_pred_val, window_size, len(X_train_sc))
         
    print(f'Epoch {(epoch)}: ,Train RSME: {train_rsme}, Test RSME: {val_rsme}')

model.load_state_dict(best_model)

plt.figure()
plt.plot(train_losses, label='training')
plt.plot(val_losses, label='validation')
plt.xlabel('Epoch');plt.ylabel('RMSE loss')
plt.axvline(x = val_losses.index(min(val_losses)), color = 'r', linestyle='dashed', label = 'lowest validation error')
plt.legend()

features_test, labels_test = timeseries_dataset_from_array(X_test_sc, window_size, horizon, label_indices=[0])
y_pred_test=model(torch.tensor(features_test).float())
plot_predictions('', X_test_sc, y_pred_test, y_pred_val, 0, len(X_test_sc))

