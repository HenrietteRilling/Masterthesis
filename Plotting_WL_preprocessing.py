# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:06:12 2023

@author: Henriette
"""

import os
import glob 
import csv
import re #package for "regular expressions", used for searching, matching and manipulating text, used for removing whitespace
import datetime as dt
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from plot_utils import cm2inch

'''Script for preprocessing and labeling Water level data from DMI
Input: uncleaned water level measurements
Output: Dataframe with preprocessed WL data, Dataframe with labels'''

#get datapath of folder with data
datafolder=r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\Data_WL'
#use glob to get all csv files in the folder
csv_files=glob.glob(os.path.join(datafolder, "*.csv"))
#specify encoding for being able to read å, ø etc.
encoding = 'utf-8'

#create list to store Wl_data and station_ids
data_list=[]
station_ids=[]
#initialize df for storing meta-information for each WL stations
WL_info_df = pd.DataFrame(columns=['id', 'location', 'name', 'unit', 'X_coordinate', 'Y_coordinate'])
#dictionary to map station number to file path
station_to_file={}
#dictionary to map station name to station number and vice versa for plotting etc.
station_name_to_id={}
station_id_to_name={}

#create list for storing information on data labeling
labels=[]

'''Preprocessing in order to read Wl-data ordered from upstream to downstream'''
#get information on order of stations, ids are sorted from river source to outlet
with open(os.path.join(datafolder, "sorted_station_ids.txt"), 'r') as file:
    station_ids=[line.strip() for line in file.readlines()] #save ids in list, strip removes \n and whitespace in strings


#map each station number to corresponding csv_file
for station_nmb in station_ids:
    for file in csv_files:
        if str(station_nmb) in file:
            station_to_file[station_nmb] = file
            break

'''Read WL-data'''
#read water level data ordered from upstream to downstream
for station_nmb in station_ids:
    file_path = station_to_file.get(station_nmb)
    #check if stationfile is in the data
    if file_path:
        #read measurements
        df=pd.read_csv(file_path, sep=';',skiprows=3, header=None, names=['date', station_nmb])
        #convert date column to pd.datetime string
        df['date']=pd.to_datetime(df.iloc[:,0],format='%Y-%m-%d %H:%M:%S')
        #set date column to index
        df.set_index('date', inplace=True)
        data_list.append(df)
        
        #read metadata (location etc.)
        with open(file_path, 'r', encoding=encoding) as file:
            reader=csv.reader(file)
            station_info=next(reader) #read the first row of each file
        #help df for concatenation
        df_station_info=pd.DataFrame({'id':[station_info[0]],'location':[station_info[1]], 'name':[station_info[2]], 'unit':[station_info[3]],'X_coordinate':[station_info[4]], 'Y_coordinate':[station_info[5]]})
        WL_info_df=pd.concat([WL_info_df, df_station_info], ignore_index=True)
        
        #map station name to id in dictionary (for plotting and accessing data)
        station_name_to_id[station_info[2].strip()]= station_nmb
        #Create inverse dictionary for the same purpose
        station_id_to_name[station_nmb]=station_info[2].strip()
        

#Sensor error is marked with -1e9, replace with nans
WL_wo_missing=[data.replace(-1e9, np.nan) for data in data_list]

# =============================================================================
# Plot showing exemplary removal of sensor error
# =============================================================================


for i, station in enumerate(data_list):
    if station.columns[0] in ('21000089'):
        df=data_list[i]
        df_clean=WL_wo_missing[i]
        fig, axes=plt.subplots(2, 1, sharex=True, figsize=cm2inch((15, 7)))
        axes[0].plot(df[(df.index>'2017-10-10')&(df.index<'2017-10-18')], label='Observations', color='blue')      
        axes[1].plot(df_clean[(df_clean.index>'2017-10-10')&(df_clean.index<'2017-10-18')], color='blue')
        axes[1].set_xlabel('Date', fontsize='large')
        #define how mane ticks are shown on the x and y axis
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        axes[0].xaxis.set_major_locator(locator)
        axes[1].xaxis.set_major_locator(locator)
        #define fontsize of ticks
        axes[1].tick_params(axis='x', labelsize='medium')
        axes[0].tick_params(axis='y', labelsize='medium')
        axes[1].tick_params(axis='y', labelsize='medium')
        #add span marking the removed data points
        axes[0].axvspan(pd.Timestamp('2017-10-15 18:00:00'), pd.Timestamp('2017-10-16 02:00:00'), alpha=0.3, color='darkgrey', label='Sensor error')
        #add legend on top of plot
        fig.legend(loc='upper center', ncol=2, fontsize='medium', frameon=False)
        fig.text(0., 0.5, 'Water level [m]', va='center', rotation='vertical', fontsize='large')
        plt.tight_layout()
        # plt.savefig(r'C:\Users\henri\Documents\Universität\Masterthesis\Report\Clean_SE.png', dpi=600)

#station Bjerringbro has a single value in 1966 and next value in 2009 (see also missingno matrix)  as this data
# is blowing up the dataframe after the hourly aggregration, we drop the first data entry.
index_BBro= station_ids.index(station_name_to_id.get('Bjerringbro'))
WL_wo_missing[index_BBro]=WL_wo_missing[index_BBro].iloc[1:, :]

# =============================================================================
# Remove spikes based on z-score
# =============================================================================

WL_wo_spikes=[]
for i, stat in enumerate(WL_wo_missing):
    station=stat.copy()
    #remove outliers based on z-score
    zscore=(station-station.mean())/station.std()
    threshold=3
    station[station.columns[0]][np.abs(zscore[station.columns[0]])>threshold]=np.nan
    WL_wo_spikes.append(station)



#______________Plot Z_scores______________________

# =============================================================================
# Plot showing exemplary removal of spike with z-score
# =============================================================================

for i, station in enumerate(WL_wo_missing):
    if station.columns[0] in ('21000089'):
        z_scores=(station-station.mean())/station.std()

        y_threshold=station.mean()+3*station.std()
        y_threshold2=station.mean()-3*station.std()
        y_mean=station.mean()
        
        df_clean=WL_wo_spikes[i]
        fig, axes=plt.subplots(2, 1, sharex=True, figsize=cm2inch((15, 7)))
        axes[0].axhline(y = y_threshold[0], color = 'lightcoral', linestyle = '--', linewidth=1, label=r'$\pm3*\sigma$')
        axes[0].axhline(y = y_threshold2[0], color = 'lightcoral', linestyle = '--', linewidth=1)
        axes[0].plot(station[(station.index>'2008-07-01')&(station.index<'2008-08-30')], label='Observation', color='blue', linestyle = 'None', marker='.', ms=2)      
        axes[1].plot(df_clean[(df_clean.index>'2008-07-01')&(df_clean.index<'2008-08-30')], color='blue', linestyle = 'None', marker='.', ms=2)      
        
        axes[1].set_xlabel('Date', fontsize='medium')
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        axes[0].xaxis.set_major_locator(locator)
        axes[1].xaxis.set_major_locator(locator)
        axes[1].tick_params(axis='x', labelsize='medium')
        axes[0].tick_params(axis='y', labelsize='medium')
        axes[1].tick_params(axis='y', labelsize='medium')
        axes[0].axvspan(pd.Timestamp('2008-07-18 20:10:00'), pd.Timestamp('2008-08-04 05:40:00'), alpha=0.3, color='darkgrey', label='Spike')
        fig.legend(loc='upper center', ncol=3, fontsize='medium', frameon=False)
        fig.text(0.02, 0.52, 'Water level [m]', va='center', rotation='vertical', fontsize='medium')
        fig.align_ylabels()
        #adjust space subplots take in window canva
        plt.subplots_adjust(bottom=0.15,top=0.9)
        #adjust space tight layout is taking in windows canva, neede that legend on top and label in bottom are shown. 
        plt.tight_layout(rect=[0.15, 0.9, 1, 1])
        plt.savefig(r'C:\Users\henri\Documents\Universität\Masterthesis\Report\Clean_spk.png', dpi=600)



# for i, station in enumerate(WL_wo_missing):
#     if station.columns[0] in ('21000089'):
#         z_scores=(station-station.mean())/station.std()
#         # plot_simple_subplots(z_scores, station_id_to_name, rows=7, cols=3)
        
#         # plt.figure()
#         # plt.plot(z_scores, color='blue', linestyle = 'None', marker='.', ms=2)
#         # plt.axhline(y = 3, color = 'r', linestyle = '-')
#         # plt.axhline(y = -3, color = 'r', linestyle = '-')
#         # plt.title(station_id_to_name.get(station.columns[0]))
#         y=station.mean()+3*station.std()
        
#         df_clean=WL_wo_spikes[i]
#         fig, axes=plt.subplots(3, 1, sharex=True, figsize=cm2inch((15, 10)))
#         axes[0].plot(z_scores[(z_scores.index>'2008-07-01')&(z_scores.index<'2008-08-30')], color='darkcyan', linestyle = 'None', marker='.', ms=2, label='Z-score')
#         axes[0].axhline(y = 3, color = 'r', linestyle = '-', linewidth=1, label='Threshold')
#         axes[0].axhline(y = -3, color = 'r', linestyle = '-', linewidth=1)
#         axes[1].axhline(y = y[0], color = 'r', linestyle = '-', linewidth=1, label='Threshold')
#         axes[1].plot(station[(station.index>'2008-07-01')&(station.index<'2008-08-30')], label='Observation', color='blue', linestyle = 'None', marker='.', ms=2)      
#         axes[2].plot(df_clean[(df_clean.index>'2008-07-01')&(df_clean.index<'2008-08-30')], color='blue', linestyle = 'None', marker='.', ms=2)      
        
#         axes[0].set_ylabel('Z-score', fontsize='medium')
#         # axes[1].set_ylabel('Water level [m]', fontsize='medium')
#         # axes[2].set_ylabel('Water level [m]', fontsize='medium')
#         axes[2].set_xlabel('Date', fontsize='medium')
#         locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
#         axes[0].xaxis.set_major_locator(locator)
#         axes[1].xaxis.set_major_locator(locator)
#         axes[2].xaxis.set_major_locator(locator)
#         axes[1].tick_params(axis='x', labelsize='medium')
#         axes[0].tick_params(axis='y', labelsize='medium')
#         axes[1].tick_params(axis='y', labelsize='medium')
#         axes[2].tick_params(axis='y', labelsize='medium')
#         axes[1].axvspan(pd.Timestamp('2008-07-18 20:10:00'), pd.Timestamp('2008-08-04 05:40:00'), alpha=0.3, color='darkgrey', label='Detected outlier')
#         fig.legend(loc='upper center', ncol=4, fontsize='medium', frameon=False)
#         fig.text(0.02, 0.37, 'Water level [m]', va='center', rotation='vertical', fontsize='medium')
#         fig.align_ylabels()
#         plt.subplots_adjust(top=0.9)
#         plt.tight_layout(rect=[0, 0.9, 1, 1])
#         # plt.savefig(r'C:\Users\henri\Documents\Universität\Masterthesis\Report\Clean_spikes.png', dpi=600)







