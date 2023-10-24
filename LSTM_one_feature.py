# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:08:08 2023

@author: Henriette
"""
# =============================================================================
# To Dos:
    # Implement maybe class Dataset??
# =============================================================================


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from Data_loader import get_WL_data, get_prcp_data
from window_data import timeseries_dataset_from_array, _get_labelled_window

from bokeh.plotting import figure, show, output_file



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

def rescale_data(data, scaler, input_dim):
    #input data has one dimension, output data 2, scaler expects the same dimensions as the input data therefore, we need to rescale
    data_reshaped=np.zeros((data.shape[0],input_dim))
    #assing data help array, reshaping tensor form dimensions (batchsize, 1,1) to (batchsize)
    data_reshaped[:,0]=data[:,0,0].numpy()    
    rescaled_data=scaler.inverse_transform(data_reshaped)
    return rescaled_data
# Eventually return only WL data of interest rescaled_data[:,0]

def get_dataloader(features, labels, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(torch.tensor(features).float(), torch.tensor(labels).float()) # insert into tensor dataset, .float() as LSTM needs torch.floa32 as input
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) # insert dataset into data loader
    return dataset, data_loader

  
def plot_predictions(epoch, y, y_pred_train, y_pred_val, train_start, train_end):
    plt.figure()
    plt.plot(y, label='observation', color='b')
    
    #Training data
    train_plot=np.ones_like(y) * np.nan
    #add another dimension
    train_plot=train_plot[:,None]
    #extract WL data and add another dimension
    y_pred_train=y_pred_train[:,0]
    train_plot[train_start:train_end]=y_pred_train[:, None]
    # train_plot[train_start:train_end]=y_pred_train[:,0]
    plt.plot(train_plot, label='train')   
    
    #validation data
    val_plot=np.ones_like(y) * np.nan
    #add another dimension
    val_plot=val_plot[:,None]
    
    y_pred_val=y_pred_val[:,0]
    val_plot[train_end+train_start:len(y)]=y_pred_val[:,None]
    plt.plot(val_plot, label='validation')
    plt.xlabel('Date'); plt.ylabel('Water level [m]')
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
#X=pd.concat([X_WL, X_prcp], axis=1).loc[X_WL.index.intersection(X_prcp.index)]

X=X_WL

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
X_val_sc = train_sc.transform(X_val)
X_test_sc = train_sc.transform(X_test)

#concat scaled timeseries for plotting
X_sc_complete=np.append(X_train_sc[:,0], X_val_sc[:,0])

'''Batching data'''
#parameter defintion
window_size=10
horizon=1

#how many features do we give as an input
nr_features=1

#get input and targets in batches with 10 timesteps input and predict the next timestep t+1, prcp data is only important for input, therefore label
features_train, labels_train = timeseries_dataset_from_array(X_train_sc, window_size, horizon, label_indices=[0])
features_val, labels_val=timeseries_dataset_from_array(X_val_sc, window_size, horizon, label_indices=[0]) 


#batch_size (int, optional) – how many samples per batch to load (default: 1)
batch_size=10

#get data_loader for all data, data_loader is an torch iterable to be able to iterate over batches
dataset_train, data_loader_train = get_dataloader(features_train, labels_train, batch_size=batch_size)
dataset_val, data_loader_val=get_dataloader(features_val, labels_val, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???


'''LSTM training + Set-Up'''
#defintion of hyperparameters
num_epochs = 50 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 1 #number of features
hidden_size = 1 #number of features in hidden state
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
train_all_losses=[]
val_losses=[]
val_all_losses=[]


#Training of the model
for epoch in range(num_epochs):
    model.train()
    running_loss_train=0.0
    for X_batch, y_batch in data_loader_train:
        #Check if there are nan values in the input ->if yes, predictions can't be calculated
        if torch.any(X_batch.isnan()):
            print('X_batch is nan!')
            
        #pytorch accumulates gradients, we need to clear them out for each instance
        optimizer.zero_grad()
        #predict
        y_pred=model(X_batch)

        #compute loss and gradients
        loss=loss_fn(y_pred, y_batch)
        #calculate MSE per batch
        running_loss_train+=loss.item()
        
        #compute gradient
        loss.backward()
        #update parameters
        optimizer.step()
    
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    #average loss per batch
    avg_train_loss=running_loss_train/len(data_loader_train)
    train_losses.append(avg_train_loss)
    
    model.eval()
    with torch.no_grad():
        '''Train data'''
        #prediction on whole training data
        y_pred_train_all = model(torch.tensor(features_train).float())
        
        train_all_mse=loss_fn(y_pred_train_all, torch.tensor(labels_train).float())
        train_all_losses.append(train_all_mse.item())

        #rescale data to actual range
        y_hat_train_all_rsc = rescale_data(y_pred_train_all, train_sc, nr_features)
        
        running_loss=0.0
        
        '''Validation data'''
        #prediction on whole validation data
        y_pred_val_all = model(torch.tensor(features_val).float())
        val_all_mse=loss_fn(y_pred_val_all, torch.tensor(labels_val).float())  
        val_all_losses.append(val_all_mse.item())
        
        #prediction on batches of validation data
        for X_val_batch, y_val_batch in data_loader_val:
            y_pred_val = model(X_val_batch)
            val_mse=loss_fn(y_pred_val,y_val_batch)
            running_loss+=val_mse.item()
        
        #calculate average loss over validation batches
        avg_val_loss=running_loss/len(data_loader_val)
        #store losses for plotting
        val_losses.append(avg_val_loss)
        
        #check if model performas better than previous models
        if avg_val_loss < best_val_loss:
            best_val_loss=avg_val_loss
            best_model=model.state_dict()

        #rescale data to actual range
        y_hat_val_all_rsc=rescale_data(y_pred_val_all, train_sc, nr_features)
        
        if epoch % 10 ==0:
            plot_predictions(epoch, X_test_all, y_hat_train_all_rsc, y_hat_val_all_rsc, window_size, len(X_train_sc))
         
    print(f'Epoch {(epoch)}: Average Train MSE: {avg_train_loss}, Average Validation MSE: {avg_val_loss}')
    print(f'All Train data MSE {train_all_mse}')
    

#load saved "best" model
model.load_state_dict(best_model)
model.eval()

#plot results of best model
y_hat_train=model(torch.tensor(features_train).float())
y_hat_val=model(torch.tensor(features_val).float())
y_hat_train=rescale_data(y_hat_train.detach(), train_sc, nr_features)
y_hat_val=rescale_data(y_hat_val.detach(), train_sc, nr_features)

plot_predictions('Best Model', X_test_all, y_hat_train, y_hat_val, window_size, len(X_train))

plt.figure()
plt.plot(train_losses, label='training (average)')
plt.plot(val_losses, label='validation (average')
plt.xlabel('Epoch');plt.ylabel('MSE loss')
plt.axvline(x = val_losses.index(min(val_losses)), color = 'r', linestyle='dashed', label = 'lowest validation error')
plt.legend()
plt.title('MSE on average per batch')


plt.figure()
plt.plot(train_all_losses, label='training')
plt.plot(val_all_losses, label='validation')
plt.xlabel('Epoch');plt.ylabel('MSE loss')
plt.axvline(x = val_all_losses.index(min(val_all_losses)), color = 'r', linestyle='dashed', label = 'lowest validation error')
plt.legend()
plt.title('MSE on all data')

# features_test, labels_test = timeseries_dataset_from_array(X_test_sc, window_size, horizon, label_indices=[0])
# y_pred_test=model(torch.tensor(features_test).float())
# plot_predictions('', X_test_sc, y_pred_test, y_pred_val, 0, len(X_test_sc))


'''Analysis of impact of loss calculation'''

# plt.figure()
# plt.plot(train_losses, label='training (average)')
# plt.plot(val_losses, label='validation (average')
# plt.plot(train_all_losses, label='training')
# plt.plot(val_all_losses, label='validation')
# plt.axvline(x = val_losses.index(min(val_losses)), color = 'r', linestyle='dashed', label = 'lowest validation error')
# plt.axvline(x = val_all_losses.index(min(val_all_losses)), color = 'r', linestyle='dashed', label = 'lowest validation error')
# plt.legend()


# a=[val_losses[val]-val_all_losses[val] for val in range(len(val_losses))]

# plt.figure()
# plt.plot(a)

# epochs=np.arange(1, num_epochs+1)
# # Create a Bokeh figure
# p = figure( x_axis_label='Epoch', y_axis_label='MSE loss')

# # Plot the value data
# p.line(epochs, train_losses, legend_label='Average training loss', line_color='blue')
# p.line(epochs, val_losses, legend_label='Average validation loss', line_color='red')
# p.line(epochs, train_all_losses, legend_label='Training loss', line_color='cyan')
# p.line(epochs, val_all_losses, legend_label='Validation loss', line_color='pink')

# # Create legend with custom labels
# p.legend.title = 'Labels'
# p.legend.label_text_font_size = '12pt'

# # Specify the output file (HTML)
# # output_file("bokeh_labeled_data.html")

# # Show the plot
# show(p)

# p=figure(x_axis_label='Epoch', y_axis_label='Difference in MSE loss')
# p.line(epochs, a)
# show(p)


# '''Plot split of the data'''
# fig, ax = plt.subplots(2,1, sharex=True)
# ax[0].plot(X_train[[test_id]], label='Train')
# ax[0].plot(X_val[[test_id]], label='Validation')
# ax[0].plot(X_test[[test_id]], label='Test')
# ax[0].set_ylabel('Water Level [m]')
# ax[0].set_title(f'Station: {test_station}')
# #ax[0].axvspan(1,1, color='green')
# ax[1].plot(X_train[test_prcp], label='Train')
# ax[1].plot(X_val[test_prcp], label='Validation')
# ax[1].plot(X_test[test_prcp], label='Test')
# ax[1].set_ylabel('Precipitation [mm]')
# ax[1].set_xlabel('Date')
# ax[1].set_title(f'Station: {test_prcp}')
# ax[1].legend()
# plt.savefig(r'C:\Users\henri\Documents\Universität\Masterthesis\First LSTM\datasplit.png', dpi=300)
