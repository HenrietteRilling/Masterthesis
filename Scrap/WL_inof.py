# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:17:24 2023

@author: Henriette
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from Data_loader import get_WL_data, get_prcp_data


#Load data
WL, _, station_name_to_id, _ = get_WL_data(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\Data_WL')
prcp=get_prcp_data(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\SVK', r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\DMI_Climate_Data_prcp', join=True)

WL_info_df=pd.DataFrame(columns=['name','start time', 'end time', 'Recorded Readings', 'Missing Records'])
WL_info_df['name']=station_name_to_id.keys()
WL_info_df.set_index('name', inplace=True)


for i, key in enumerate(station_name_to_id.keys()):
    idx=station_name_to_id.get(key)
    X=WL[[idx]]
      
    #make sure that only period in which sensor data is available is used
    X=X[(X.index>X.first_valid_index())&(X.index<X.last_valid_index())]
    print(len(X)-X.isna().sum())
    WL_info_df.loc[[key],['Recorded Readings']]=(len(X)-X.isna().sum())[0]
    WL_info_df.loc[[key],['Missing Records']]=X.isna().sum()[0]
    WL_info_df.loc[[key],['start time']]=X.first_valid_index().date().strftime('%d-%m-%Y')
    WL_info_df.loc[[key],['end time']]=X.last_valid_index().date().strftime('%d-%m-%Y')

print(WL_info_df.head())