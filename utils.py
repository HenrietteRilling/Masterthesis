# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:37:50 2023

@author: Henriette
"""

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
    data_reshaped[:,0]=data[:,0,0].numpy() #for tensor: add .numpy()  
    rescaled_data=scaler.inverse_transform(data_reshaped)
    return rescaled_data
# Eventually return only WL data of interest rescaled_data[:,0]


#unscale for feed forwards network, output is has different sizes
def rescale_data_ffn(data, scaler, input_dim):
    # import pdb
    # pdb.set_trace()
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

def timeseries_dataset_from_array(data, window_size, horizon, stride=1, label_indices: list=None):
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