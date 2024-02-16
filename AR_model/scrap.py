# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:10:41 2023

@author: Henriette
"""


import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime
from utils import get_WL_data, get_prcp_data, get_test_data, load_data
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
test_station='ns Uldumkær'
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
ax1.axhline(y=49.7, color='red')
# Create a second y-axis for precipitation
ax2 = ax1.twinx()
ax2.plot(-X[test_prcp], color='green', label='Precipitation')
ax2.set_ylabel('Precipitation', color='green')
ax2.tick_params('y', colors='green')
ax2.set_ylim(-20, 0)
# Invert the tick labels on the second y-axis such that they are displayed positive

yticks = ax2.get_yticks()
ax2.set_yticks(yticks)
ax2.set_yticklabels([abs(y) for y in yticks])



# '''
# Old function for plotting directly in simulation
# '''
# def get_best_weigth_paths(path, nr_of_models):
#     '''
#     The same model configuration is trained for n times for balancing statistical variations. 
#     Visualisation of results is based on model with lowest validation error, this function identifies
#     the model with the lowest valdiation error among the model runs.
#     '''
#     losslogpaths=glob.glob(os.path.join(path, '*losslog*.csv'))
#     #define regular expression pattern for extracting training horizon of weight file name
#     pattern=re.compile(r'losslog_(\d+)_\d+\.csv')
#     #sort weights from shortest to longest training horizon
#     losslogpaths=sorted(losslogpaths, key=lambda x: int(re.search(pattern, x).group(1)))
#     best_model_paths=[]
#     #loop over lossfiles and find for each training horizon the model with the lowest validation error
#     for i, losspath in enumerate(losslogpaths):
#         #read very last validation loss from file
#         if i%nr_of_models==0:
#             val_loss=1.0
#             best_model_paths.append(losspath)
        
#         #read only value of last row as this corresponds to the lowest validation error
#         cur_val_loss=pd.read_csv(losspath, header=None, sep=';',usecols=[1],skiprows=lambda x: x < sum(1 for line in open(losspath)) - 1).iloc[0,0]
#         if cur_val_loss<val_loss:
#             val_loss=cur_val_loss
#             #replace loss-expressions with pattern of weightstring
#             best_model_paths[-1]=losspath.replace('losslog', 'weights').replace('csv', 'pth')    
                
#     return best_model_paths


#     # for i, conf in enumerate(configs_tested):
#     #     weight_paths=get_best_weigth_paths(conf[0],config['n_models'])
#     #     plot_imputation(X, test_id, config['prcp_station'], conf[0], config['train_period'], config['test_period'], config['plot_horizon'], 
#     #                     conf[3], conf[4], conf[5], conf[2], weight_paths) 
 


def plot_split_ofdata():
    #select test stations and extract data
    test_station='ns Uldumkær'
    test_prcp='05225'
    
    X, test_id=load_data(test_station, test_prcp)
    
    dates=pd.to_datetime(X.index)
    
    X_train_plot=X.copy()
    X_val_plot=X.copy()
    X_test_plot=X.copy()
    X_train_plot['2019-01-01':]=np.nan
    X_val_plot[(dates<'2019-01-01')]=np.nan
    X_val_plot[(dates>'2021-12-31')]=np.nan
    X_test_plot[dates<'2022-01-01']=np.nan
    
    fig, ax=plt.subplots(1,1, figsize=cm2inch((15,7)))
    ax.plot(dates, X_train_plot[test_id], label='Train')
    ax.plot(dates, X_val_plot[test_id], label='Validation')
    ax.plot(dates,X_test_plot[test_id], label='Test')
    
    #adjust xticks
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    
    #set fontzise of yaxis
    ax.tick_params('y', labelsize='medium')
    ax.set_ylim(49.32, 51.1)               
    # Create a second y-axis for precipitation
    ax2 = ax.twinx()
    # ax2.plot(dates, X[test_prcp])
    ax2.bar(dates, -X[test_prcp], width=0.01, color='lightsteelblue',edgecolor='lightsteelblue') #alternative: cornflowerblue
    ax2.tick_params('y', labelsize='medium')
    ax2.set_ylim(-50, 0)
    
    # Invert the tick labels on the second y-axis such that they are displayed positive
    yticks = ax2.get_yticks()
    ax2.set_yticks(yticks)
    #make yticks label positive and without decimals
    ax2.set_yticklabels([f'{abs(y):.0f}' for y in yticks])
    
    ax.set_xlabel('Date', fontsize='large')
    ax.set_ylabel('Water level [m]', fontsize='large')
    # ax2.set_ylabel('Precipitation [mm/h]', fontsize='large', rotation=-90)
    
    fig.legend(loc='upper center', ncol=4, fontsize='medium', frameon=False)
    fig.text(0.96, 0.5, 'Precipitation [mm/h]', va='center',rotation=-90, fontsize='large')
    plt.subplots_adjust(left= 0.0, bottom=0.06,right=0.96, top=1.0, hspace=0.2)
    # #adjust space tight layout is taking in windows canva, neede that legend on top and label in bottom are shown. 
    plt.tight_layout(rect=[0.0, 0.06 ,0.96, 0.9],pad=0.3) #rect: [left, bottom, right, top]
    plt.savefig(r'C:\Users\henri\Documents\Universität\Masterthesis\Report\model_data.png', dpi=600)
    
    
#plot split for 2station model
def plot_split_ofdata2():    
    station=['ns Uldumkær','Brestenbro']
    fig, axes=plt.subplots(2,1, figsize=cm2inch((15,9)), sharex=True)
    axes=axes.flatten()
    linestyles=['solid', 'dashed']


    for i, test_station in enumerate(station):
        ax=axes[i]
        test_prcp='05225'
        # import pdb
        # pdb.set_trace()
        X, test_id=load_data(test_station, test_prcp)
        
        dates=pd.to_datetime(X.index)
        
        X_train_plot=X.copy()
        X_val_plot=X.copy()
        X_test_plot=X.copy()
        X_train_plot['2018-11-01':]=np.nan
        X_train_plot[(dates<'2015-11-01')]=np.nan
        X_val_plot[(dates<'2018-11-01')]=np.nan
        X_val_plot[(dates>'2020-10-31')]=np.nan
        X_test_plot[dates<'2020-10-31']=np.nan
        X_test_plot[dates>'2021-11-01']=np.nan
        
        if i==0:
            ax.plot(dates, X_train_plot[test_id], label='Train', linestyle='solid', color='blue')
            ax.plot(dates, X_val_plot[test_id], label='Validation', linestyle='solid', color='darkorange')
            ax.plot(dates,X_test_plot[test_id], label='Test', linestyle='solid',color='green')
        else:
            ax.plot(dates, X_train_plot[test_id], linestyle='solid', color='blue')
            ax.plot(dates, X_val_plot[test_id],  linestyle='solid',color='darkorange')
            ax.plot(dates,X_test_plot[test_id],  linestyle='solid', color='green')
            
        #adjust xticks
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        #set fontzise of yaxis
        ax.tick_params('y', labelsize='medium')
        # ax.set_ylim(49.32, 51.1)               
        # Create a second y-axis for precipitation
        ax2 = ax.twinx()
        # ax2.plot(dates, X[test_prcp])
        ax2.bar(X['2015-11-01':'2021-11-01'].index, -X['2015-11-01':'2021-11-01'][test_prcp], width=0.01, color='lightsteelblue',edgecolor='lightsteelblue') #alternative: cornflowerblue
        ax2.tick_params('y', labelsize='medium')
        ax2.set_ylim(-50, 0)
        
        # Invert the tick labels on the second y-axis such that they are displayed positive
        yticks = ax2.get_yticks()
        ax2.set_yticks(yticks)
        #make yticks label positive and without decimals
        ax2.set_yticklabels([f'{abs(y):.0f}' for y in yticks])
        ax.set_title(f'{test_station}', loc='left', fontsize='medium')

    ax.set_xlabel('Date', fontsize='large')
    fig.legend(loc='upper center', ncol=3, fontsize='medium', frameon=False)
    fig.text(0.96, 0.5, 'Precipitation [mm/h]', va='center',rotation=-90, fontsize='large')
    fig.text(0.02, 0.5, 'Water level [m]', va='center', rotation='vertical', fontsize='large')
    plt.subplots_adjust(left= 0.06, bottom=0.06,right=0.96, top=1.0, hspace=0.2)
    # #adjust space tight layout is taking in windows canva, neede that legend on top and label in bottom are shown. 
    plt.tight_layout(rect=[0.06, 0.06 ,0.96, 0.9],pad=0.3) #rect: [left, bottom, right, top]
    # plt.savefig(r'C:\Users\henri\Documents\Universität\Masterthesis\Report\model_data_LSTM_AW.png', dpi=600)
    
plot_split_ofdata2()