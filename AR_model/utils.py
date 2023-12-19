# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:37:50 2023

@author: Henriette
"""

import os
import pandas as pd
import pickle
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def scale_data(data):
    scaler=MinMaxScaler()
    scaled_data=scaler.fit_transform(data)
    return scaled_data, scaler

def rescale_data(data, scaler, input_dim):
    #input data has one dimension, output data 2, scaler expects the same dimensions as the input data therefore, we need to rescale
    data_reshaped=np.zeros((data.shape[0],input_dim))
    #assing data to help array, reshaping tensor form dimensions (batchsize, 1,1) to (batchsize)
    data_reshaped[:,0]=data[:,0,0] #for tensor: add .numpy()  
    rescaled_data=scaler.inverse_transform(data_reshaped)
    return rescaled_data
# Eventually return only WL data of interest rescaled_data[:,0]


#unscale for feed forwards network, output is has different sizes
def rescale_data_ffn(data, scaler, input_dim):
    import pdb
    pdb.set_trace()
    #input data has one dimension, output data 2, scaler expects the same dimensions as the input data therefore, we need to rescale
    data_reshaped=np.zeros((data.shape[1],input_dim))
    #assing data to help array, reshaping tensor form dimensions (batchsize, 1,1) to (batchsize)
    data_reshaped[:,0]=data[0,:,0]    
    nanidx=np.isnan(data_reshaped)
    rescaled_data=scaler.inverse_transform(data_reshaped)
    rescaled_data[nanidx]=np.nan
    return rescaled_data[:,0]
# Eventually return only WL data of interest rescaled_data[:,0]





def _get_labelled_window(windowed_data, horizon: int):
    """Create labels for windowed dataset
    Input: [0, 1, 2, 3, 4, 5] and horizon=1
    Output: ([0, 1, 2, 3, 4], [5])

    Parameters
    ----------
    data : array
        time series to be labelled
    horizon : int
        the horizon to predict
    """
    return windowed_data[:, :-horizon], windowed_data[:, -horizon:]

def _get_labelled_window_AR(windowed_data, horizon: int):
    """Create labels for windowed dataset
    if horizon=1 and FFNN:
    Input: [0, 1, 2, 3, 4, 5] and horizon=1
    Output: ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
    
    if horizon>1 and LSTM:
        Input: [0, 1, 2, 3, 4, 5, 6, 7, 8] and horizon=3
        Output: ([0,1,2,3,4,5,6,7],[6,7,8])

    Parameters
    ----------
    data : array
        time series to be labelled
    horizon : int
        the horizon to predict
    """
    
    if horizon>1:
        return windowed_data[:, :-1], windowed_data[:, -horizon:]
    
    else: return windowed_data[:, :-horizon], windowed_data[:, horizon:] 


def timeseries_dataset_from_array(data, window_size, horizon, stride=1, label_indices: list=None, AR=False):
    # Adapted from https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
    # and https://www.mlq.ai/time-series-tensorflow-windows-horizons/
    """Creates windows and labels
    Input data must have format [time, ..., features], where ... can be e.g. lat and lon.
    Outputs have format [batch, time, ..., feature]. Number of features can vary depending on label_indices.

    Returns
    -------
    tuple(array, array)
        Windows and labels with shapes [batch, time, ..., feature]
    """

    # Create window of the specific size. Add horizon for to include labels.
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
    # Create the timesteps. subtract window_size and horizon to get equal length windows and subtract 1 to account for
    # 0-indexing
    time_step = np.expand_dims(
        np.arange(data.shape[0] - (window_size + horizon - 1), step=stride), axis=0
    ).T

    # Create the window indexex
    window_indexes = window_step + time_step

    # Get the windows from the data in [batch, time, ..., features]
    windowed_data = data[window_indexes]
    

    # Split windows and labels
    if AR == True:
        windows, labels = _get_labelled_window_AR(windowed_data, horizon)
    
    else: 
        windows, labels = _get_labelled_window(windowed_data, horizon)

    
    # Select only the labels we need
    if label_indices is not None:
        assert (
            type(label_indices) == list
        ), f"label_indices needs to be list[int], but is {type(label_indices)}"
        labels = labels[..., label_indices]

    return windows, labels

def get_dataloader(features, labels, batch_size, shuffle=True):
    dataset = torch.utils.data.TensorDataset(torch.tensor(features).float(), torch.tensor(labels).float()) # insert into tensor dataset, .float() as LSTM needs torch.floa32 as input
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) # insert dataset into data loader
    return dataset, data_loader


'''Function takes datapath as input and returns preprocessed DataFrame with Water Level data, Dataframe with labels 
and dictionaries linking ids + names'''
def get_WL_data(datapath):
    #load WL data with labels
    data=pd.read_csv(os.path.join(datapath, 'WL_w_labels.csv'), sep=',')
    #convert date column to pd.datetime string
    data['date']=pd.to_datetime(data.iloc[:,0],format='%Y-%m-%d %H:%M:%S')
    #set date column to index
    data.set_index('date', inplace=True)
    
    #load dictionaries with information on station names
    with open(os.path.join(datapath, 'station_name_to_id.pkl'), 'rb') as file:
        station_name_to_id=pickle.load(file)
    
    with open(os.path.join(datapath, 'station_id_to_name.pkl'), 'rb') as file:
        station_id_to_name=pickle.load(file)
    
    
    #dataframe contains both the WL data as well as the labels, let's split the data in 2 separate dataframes
    #WL data is in the first 21 columns, lables in the remaining 21 columns
    WL=data.iloc[:, :len(station_name_to_id)].copy()
    labels=data.iloc[:, len(station_name_to_id):].copy()
    #rename label columns as their name changed while loading
    labels.columns=WL.columns
    
    return WL, labels, station_name_to_id, station_id_to_name

'''Function takes datapath as input and returns preprocessed DataFrame with prcp data'''

def get_prcp_data(SVK_datapath, DMI_datapath, join=False):
    
    #load preprocessed SVK data
    df_SVK=pd.read_csv(os.path.join(SVK_datapath, 'SVK_preprocessed.csv'), delimiter=',')
    # Convert the 'time' column to datetime
    df_SVK['time']=pd.to_datetime(df_SVK.iloc[:,0],format='%Y-%m-%d %H:%M:%S')
    
    # Set the 'time' column as the index
    df_SVK.set_index('time', inplace=True)
 
    #load preprocessed DMI data
    df_DMI=pd.read_csv(os.path.join(DMI_datapath, 'DMI_preprocessed.csv'), delimiter=',')
    # Convert the 'date' column to datetime
    df_DMI['date']=pd.to_datetime(df_DMI.iloc[:,0],format='%Y-%m-%d %H:%M:%S')
    
    # Set the 'date' column as the index
    df_DMI.set_index('date', inplace=True)
    
    #Remove time zone information, other data doesn't hold timezone information, timezone information causes issue with concating etc.
    df_DMI.index=df_DMI.index.tz_localize(None)
    
    if join:
        #concat dataframes to one dataframe
        return pd.concat([df_SVK, df_DMI], axis=1)
    
    else:    
        return df_SVK, df_DMI


def get_test_data(stat_id, data_df, interpolation=True):
    '''get data for specific station (e.g. WL, precipitation), cropped to their actual timeframe
    data_df: dataframe, timeseries data with  station_ids as column names'''
    
    X=data_df[[stat_id]]
    
    #make sure that only period in which sensor data is available is used
    X=X[(X.index>X.first_valid_index())&(X.index<X.last_valid_index())]
    #interpolate missing values
    if interpolation:
        X.interpolate(inplace=True)
    return X