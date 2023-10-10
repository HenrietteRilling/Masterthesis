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
X_test=X['2019-01-01':'2021-12-31']
X_val=X['2022-01-01':'2022-12-31']

#scale and normalise such that all data has a range between [0,1]
mm=MinMaxScaler()
X_train_mm=mm.fit_transform(X_train)
X_test_mm=mm.fit_transform(X_test)
X_val_mm=mm.fit_transform(X_val)

'''Batching data'''
#parameter defintion
window_size=10
horizon=1

#get input and targets in batches with 10 timesteps input and predict the next timestep t+1, prcp data is only important for input, therefore label
features, labels = timeseries_dataset_from_array(X_train_mm, window_size, horizon, label_indices=[0]) 
dataset = torch.utils.data.TensorDataset(torch.tensor(features), torch.tensor(labels)) # insert into tensor dataset
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True) # insert dataset into data loader
#batch_size (int, optional) – how many samples per batch to load (default: 1)

'''LSTM training + SetUp'''
#defintion of hyperparameters
num_epochs = 3 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 2 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

#Instantiate  the class LSTM object.
model = LSTM(input_size, hidden_size, num_layers) #our lstm class

#Choose loss function and Optimizer
loss_fn = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

#Training of the model
for epoch in range(num_epochs):
    print(f'Epoch: {epoch}')
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
        print(f'Loss: {loss.item()}')
        loss.backward()
        optimizer.step()
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
