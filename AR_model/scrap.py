# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:10:41 2023

@author: Henriette
"""


import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime
from utils import get_WL_data, get_prcp_data, get_test_data
from plot_utils import cm2inch


#############################################
#load data
WL, _, station_name_to_id, _ = get_WL_data(r'./Data')
prcp=get_prcp_data(r'./Data', r'./Data', join=True)

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
# X=X_WL 

#################################################
#split data in 70% train, 20% val, 10% test
#note: pd slicing is inlcusive
X_train=X['2012-01-01':'2018-12-31']
X_val=X['2019-01-01':'2021-12-31']
X_test=X['2022-01-01':'2022-12-31']


fig, ax1 = plt.subplots()

# Plot water level on the bottom axis
ax1.plot(X[test_id], color='blue', label='Water Level')
ax1.set_xlabel('Date')
ax1.set_ylabel('Water Level', color='blue')
ax1.tick_params('y', colors='blue')

# Create a second y-axis for precipitation
ax2 = ax1.twinx()
ax2.plot(-X[test_prcp], color='green', label='Precipitation')
ax2.set_ylabel('Precipitation', color='green')
ax2.tick_params('y', colors='green')

