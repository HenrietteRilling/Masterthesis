# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:40:50 2023

@author: Henriette
"""

import os
import pandas as pd
import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

'''Function takes datapaht as input and returns preprocessed DataFrame with prcp data'''

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

#prcp=get_prcp_data(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\SVK', r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\DMI_Climate_Data_prcp', join=True)
#SVK, DMI=get_prcp_data(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\SVK', r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\DMI_Climate_Data_prcp')