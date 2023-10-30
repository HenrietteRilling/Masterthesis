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
        
        #initialise self and hidden state as None
        self.hidden_state=None
        self.cell_state=None
     
    def forward(self, x, future_pred=0):
        import pdb
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm1(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.fc(hn) #Final output

        out_future=[]
        for i in range(future_pred):
            # pdb.set_trace()
            # if self.hidden_state==None:
            # h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32) #hidden state
            # c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32) #internal state        
            # else:
                # h_0=self.hidden_state
                # c_0=self.cell_state
            #shift observations by one timestep
            x_shifted=torch.zeros(x.size())
            x_shifted[:,:-1,:]=x[:,1:,:]
            #add y_hat as newest observation
            x_shifted[:,-1,:]=out #alternativ: x_shifte[:,-1:,:]
            #create future predictions if future_pred>0 is passed
            #same as forward step above, using last output/prediction as input            
            output, (hn, cn)=self.lstm1(x_shifted, (h_0, c_0))
            hn_view=hn.view(-1, self.hidden_size)
            out=self.fc(hn_view)
            #To Do
            # if i%10 == 0:
            #     if i==0:
            #         plt.figure()
            #     plt.plot(out[:,0].detach(), label=f'prediction future={i}')
            #     plt.legend()
            #To Do: save outputs for plotting and as future predicitons
            out_future.append(out.unsqueeze(-1))
            # self.hidden_state, self.cell_state = hn, cn

        if future_pred == 0:
            return out.unsqueeze(-1) #unsqueeze adds another dimension of 1 to the tensor, necessary to have same shape as batched target data   
        else:
            # pdb.set_trace()
            return out_future #return only future predictions


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
    #assing data to help array, reshaping tensor form dimensions (batchsize, 1,1) to (batchsize)
    data_reshaped[:,0]=data[:,0,0].numpy()    
    rescaled_data=scaler.inverse_transform(data_reshaped)
    return rescaled_data
# Eventually return only WL data of interest rescaled_data[:,0]

def get_dataloader(features, labels, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(torch.tensor(features).float(), torch.tensor(labels).float()) # insert into tensor dataset, .float() as LSTM needs torch.floa32 as input
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) # insert dataset into data loader
    return dataset, data_loader


def train_one_epoch(epoch, model, train_dataloader, loss_func, optimizer):
    #assure that model is in training mode
    model.train()
    running_loss=0.0
    
    for X_batch, y_batch in train_dataloader:
        #pytorch accumulates gradients, therefore, clear gradients in each instance
        optimizer.zero_grad()
        #predict
        y_hat=model(X_batch)
        #Compute loss and gradients
        loss=loss_func(y_hat, y_batch)
        #compute gradient
        loss.backward()
        #update parameters
        optimizer.step()
    running_loss+=loss.item()
    print(f"Epoch: {epoch}, Train loss: {running_loss:>4f}")
    return running_loss 

def eval_one_epoch(epoch, model, val_dataloader, loss_func):
    model.eval()
    running_loss=0.0

    with torch.no_grad():
        for X_batch, y_batch in val_dataloader:
            y_hat = model(X_batch)
            loss=loss_func(y_hat,y_batch)
            running_loss+=loss.item()
            
    avg_loss=running_loss/len(val_dataloader)
    print(f'Epoch: {epoch}, Val loss: {avg_loss:>4f}')
    return avg_loss
  
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
# X=pd.concat([X_WL, X_prcp], axis=1).loc[X_WL.index.intersection(X_prcp.index)]
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
num_epochs = 10 #1000 epochs
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
val_losses=[]

for epoch in range(num_epochs):
    #model training
    loss=train_one_epoch(epoch, model, data_loader_train, loss_fn, optimizer)
    #keep track of losses
    train_losses.append(loss)
    
    #model validation
    loss=eval_one_epoch(epoch, model, data_loader_val, loss_fn)
    val_losses.append(loss)
    
    #check if model performs better than previous models
    if loss < best_val_loss:
        best_val_loss=loss
        best_model=model.state_dict()


#plot development of losses   
plt.figure()
plt.plot(train_losses, label='training')
plt.plot(val_losses, label='validation')
plt.xlabel('Epoch');plt.ylabel('MSE loss')
plt.axvline(x = val_losses.index(min(val_losses)), color = 'r', linestyle='dashed', label = 'lowest validation error')
plt.legend()
plt.title('MSE on average per batch')


#load saved "best" model #To Do
model.load_state_dict(best_model)
model.eval()
future=50
pred=model(torch.tensor(features_val).float(), future_pred=future)


pred=[rescale_data(p.detach(), train_sc, nr_features) for p in pred]

pred_fut=[]
for i in range(future):
    pred_fut.append(pred[i][-1])

fut_plot1=np.ones(len(X_val)+future) * np.nan
fut_plot2=np.ones(len(X_val)+future) * np.nan
fut_plot1[:len(X_val)]=X_val[test_id].to_numpy()
fut_plot2[len(X_val):]=pred_fut

plt.figure()
# plt.plot(fut_plot)
plt.plot(fut_plot1, label='validation')
plt.plot(fut_plot2, label='future') #To DOOOOOOO
plt.legend()
plt.title('Validation with future predictions')


plt.figure()
plt.plot(pred_fut)
plt.title('Future predictions only')

plt.figure()
plt.plot(pred[0])
plt.title('Prediction at fut=1')


#plot results of best model
y_hat_train=model(torch.tensor(features_train).float())
y_hat_val=model(torch.tensor(features_val).float())
y_hat_train=rescale_data(y_hat_train.detach(), train_sc, nr_features)
y_hat_val=rescale_data(y_hat_val.detach(), train_sc, nr_features)

plot_predictions('Best Model', X_test_all, y_hat_train, y_hat_val, window_size, len(X_train))

plt.figure()
for i, p in enumerate(pred):
    if i%5==0:
        plt.plot(p, label=f'Prediction {i}')
        plt.legend()
plt.title('Best model predictions')
    
# # features_test, labels_test = timeseries_dataset_from_array(X_test_sc, window_size, horizon, label_indices=[0])
# # y_pred_test=model(torch.tensor(features_test).float())
# # plot_predictions('', X_test_sc, y_pred_test, y_pred_val, 0, len(X_test_sc))


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
