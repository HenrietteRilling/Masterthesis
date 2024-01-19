# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:33:39 2024

@author: Henriette
"""

'''Plotting to test why PI for training horizon 1 is so bad'''

import os
import pickle
import csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta
from plot_utils import cm2inch





# respath=r'C:\Users\henri\Desktop\LSTM_preliminary'
respath=r'C:\Users\henri\Documents\Universität\Masterthesis\Results\LSTM_AR'
configpath=os.path.join(respath, 'configs.csv')
#Read csv file with model configurations
with open(configpath, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Convert the CSV rows to a list
    config_list = list(csv_reader)


#Constants
plot_h=[1]
train_h=[1]
window=[10, 20, 50]

#plot statics
msize=1
colorlist=['darkorange', 'lightskyblue', 'lime', 'olive', 'darkviolet']   


axidx=-1

for config in config_list[:1]:
    #get path of pkl file for current model configuration
    pred_pkl_path=os.path.join(respath, f'{os.path.basename(config[0])}.pkl')
    #Read pickle
    if os.path.exists(pred_pkl_path):
        axidx+=1
        with open(pred_pkl_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
    else: continue 
    
    #Extract data
    pred_list=data[0]
    X_test=data[1]
    
    #Get id of test station and precipitation station
    test_id=config[-2]
    test_prcp=config[-1]   
    #one figure per configuration

X_test_rain_array=X_test[[test_prcp]].to_numpy()

no_prcp=False
if no_prcp:
    respath=r'C:\Users\henri\Desktop\LSTM_prelim_2'
    configpath=os.path.join(respath, 'configs.csv')
    #Read csv file with model configurations
    with open(configpath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Convert the CSV rows to a list
        config_list = list(csv_reader)
    
    
    for config in config_list[:1]:
        #get path of pkl file for current model configuration
        pred_pkl_path=os.path.join(respath, f'{os.path.basename(config[0])}.pkl')
        #Read pickle
        if os.path.exists(pred_pkl_path):
            axidx+=1
            with open(pred_pkl_path, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
        else: continue 
        
        #Extract data
        pred_list=data[0]
        X_test=data[1]
        
        #Get id of test station and precipitation station
        test_id=config[-2]  
        #one figure per configuration
    
preds=pred_list[0]
#calculate error of 1-step prediction model results
X_test_array=X_test[[test_id]].to_numpy()

#negative error=oversetimation
error_preds=X_test_array[1:]-preds[1:,:1]
error_prevobs=X_test_array[1:]-X_test_array[:-1]
dates=pd.to_datetime(X_test.index)


# =============================================================================
# Residual plots
# =============================================================================
#residuals over time
plt.figure()
plt.plot(dates[1:],error_preds, label='Prediction', color='darkorange', linestyle='None', marker='.', ms=msize)
plt.plot(dates[1:],error_prevobs, label='Previous value',color='green', linestyle='None', marker='.', ms=msize)
plt.legend()
plt.ylabel('Residual')
plt.xlabel('Date')

#resdiuals over time with precipitation
fig, ax =plt.subplots(1,1)
ax.plot(dates[1:],error_preds, label='Prediction', color='darkorange', linestyle='None', marker='.', ms=msize)
ax.plot(dates[1:],error_prevobs, label='Previous value',color='green', linestyle='None', marker='.', ms=msize)
ax.legend(loc='lower left')
ax.set_ylabel('Residual')
ax.set_xlabel('Date')

ax1=ax.twinx()
ax1.bar(dates[1:], -X_test_rain_array[1:, 0], width=0.05, color='lightsteelblue',edgecolor='black') #alternative: cornflowerblue
# Set the y-axis tick labels to display without decimal precision
ax1.tick_params('y', labelsize='medium')
ax1.set_ylim(-40, 0)

# Invert the tick labels on the second y-axis such that they are displayed positive
yticks = ax1.get_yticks()
ax1.set_yticks(yticks)
#make yticks label positive and without decimals
ax1.set_yticklabels([f'{abs(y):.0f}' for y in yticks])
ax1.set_ylabel('Precipitation [mm/h]', rotation=-90, labelpad=11)


#Errorbars over time
plt.figure()
plt.errorbar(dates[1:],X_test_array[1:,0], yerr=np.abs(error_preds[:,0]), ecolor='darkorange',fmt='None', elinewidth=0.5, label='Prediction', linestyle='None', marker='.', ms=msize)
plt.errorbar(dates[1:],X_test_array[1:,0], yerr=np.abs(error_prevobs[:,0]), ecolor='green', fmt='None',elinewidth=0.5, label='Previous observation', linestyle='None', marker='.', ms=msize)
# plt.plot(dates[1:],error_prevobs, label='Previous value', linestyle='None', marker='.', ms=msize)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Water level [m]')

#Predictions, observations and previous value as predictions in one plot
plt.figure()
plt.plot(dates[1:], X_test_array[1:], color='blue', label='Observation', linestyle='-', marker='.', ms=msize)
plt.plot(dates[1:], preds[1:,:1], color='darkorange', label='Predictions', linestyle='None', marker='.', ms=msize)
plt.plot(dates[1:], X_test_array[:-1], color='green', label='Previous observation', linestyle='None', marker='.', ms=msize)
plt.xlabel('Date')
plt.ylabel('Water level [m]')
plt.legend()


# =============================================================================
# Scatterplots
# =============================================================================
#residuals against observations
plt.figure()
plt.scatter(X_test_array[1:], error_preds, s=msize, c='darkorange',label='preds')
plt.scatter(X_test_array[1:], error_prevobs, s=msize, c='green',label='prevobs')
plt.legend()
plt.xlabel('Observed water level [m]')
plt.ylabel('Residual')

#residuals against precipitation
plt.figure()
plt.scatter(X_test_rain_array[1:], error_preds, s=msize, c='darkorange',label='preds')
plt.scatter(X_test_rain_array[1:], error_prevobs, s=msize, c='green',label='prevobs')
plt.legend()
plt.xlabel('Precipitation [mm/h]')
plt.ylabel('Residual')



# =============================================================================
# Subplots in variations
# =============================================================================

#Residuals, errorbars, plottet values

fig, ax=plt.subplots(3, 1, sharex=True)

ax1=ax[0]
ax2=ax[1]
ax3=ax[2]
ax1.plot(dates[1:],error_preds,  linestyle='None', marker='.', ms=msize, color='darkorange')
ax1.plot(dates[1:],error_prevobs, linestyle='None', marker='.', ms=msize, color='green')
# ax1.legend()
ax1.set_ylabel('Residual')

ax2.errorbar(dates[1:],X_test_array[1:,0], yerr=np.abs(error_preds[:,0]), fmt='None', color='blue', ecolor='darkorange', linestyle='None', marker='.', ms=msize)
ax2.errorbar(dates[1:],X_test_array[1:,0], yerr=np.abs(error_prevobs[:,0]),fmt='None',color='blue', ecolor='green', linestyle='None', marker='.', ms=msize)
# plt.plot(dates[1:],error_prevobs, label='Previous value', linestyle='None', marker='.', ms=msize)
ax2.set_ylabel('WL [m]')
# ax2.legend()

ax3.plot(dates[1:], X_test_array[1:], color='blue', label='Observation', linestyle='None', marker='.', ms=msize)
ax3.plot(dates[1:], preds[1:,:1], color='darkorange', label='Predictions', linestyle='None', marker='.', ms=msize)
ax3.plot(dates[1:], X_test_array[:-1], color='green', label='Previous observation', linestyle='None', marker='.', ms=msize)
ax3.set_xlabel('Date')
ax3.set_ylabel('WL [m]')
fig.legend(loc='upper center', ncol=3, fontsize='medium', frameon=False)

#Residuals over time, values over time with precipitation

fig, ax=plt.subplots(2, 1, sharex=True)

ax1=ax[0]
ax2=ax[1]
ax1.plot(dates[1:],error_preds,  linestyle='None', marker='.', ms=msize, color='darkorange')
ax1.plot(dates[1:],error_prevobs, linestyle='None', marker='.', ms=msize, color='green')
# ax1.legend()
ax1.set_ylabel('Residual')

ax2.plot(dates[1:], X_test_array[1:], color='blue', label='Observation', linestyle='None', marker='.', ms=msize)
ax2.plot(dates[1:], preds[1:,:1], color='darkorange', label='Predictions', linestyle='None', marker='.', ms=msize)
ax2.plot(dates[1:], X_test_array[:-1], color='green', label='Previous observation', linestyle='None', marker='.', ms=msize)
ax2.set_xlabel('Date')
ax2.set_ylabel('WL [m]')
fig.legend(loc='upper center', ncol=3, fontsize='medium', frameon=False)

ax3 =ax1.twinx()
ax3.bar(dates[1:], -X_test_rain_array[1:, 0], width=0.05, color='lightsteelblue',edgecolor='black') #alternative: cornflowerblue
# Set the y-axis tick labels to display without decimal precision
ax3.tick_params('y', labelsize='medium')
ax3.set_ylim(-40, 0)

# Invert the tick labels on the second y-axis such that they are displayed positive
yticks = ax3.get_yticks()
ax3.set_yticks(yticks)
#make yticks label positive and without decimals
ax3.set_yticklabels([f'{abs(y):.0f}' for y in yticks])
ax3.set_ylabel('Precipitation [mm/h]')



#zoomed plot
start_date=pd.to_datetime('2022-06-04')
end_date=pd.to_datetime('2022-06-13')


fig, ax=plt.subplots(2, 1, sharex=True)

ax1=ax[0]
ax2=ax[1]
ax1.plot(dates[1:],error_preds,  linestyle='None', marker='.', ms=msize, color='darkorange')
ax1.plot(dates[1:],error_prevobs, linestyle='None', marker='.', ms=msize, color='green')
# ax1.legend()
ax1.set_ylabel('Residual')
ax1.set_xlim(start_date, end_date)

ax2.plot(dates[1:], X_test_array[1:], color='blue', label='Observation', linestyle='None', marker='.', ms=msize)
ax2.plot(dates[1:], preds[1:,:1], color='darkorange', label='Predictions', linestyle='None', marker='.', ms=msize)
ax2.plot(dates[1:], X_test_array[:-1], color='green', label='Previous observation', linestyle='None', marker='.', ms=msize)
ax2.set_xlabel('Date')
ax2.set_ylabel('WL [m]')
ax2.set_xlim(start_date, end_date)
ax2.set_ylim(49.6, 50.25 )
plt.xticks(rotation=25)
fig.legend(loc='upper center', ncol=3, fontsize='medium', frameon=False, markerscale=6)


ax3 =ax1.twinx()
ax3.bar(dates[1:], -X_test_rain_array[1:, 0], width=0.05, color='lightsteelblue',edgecolor='black') #alternative: cornflowerblue
# Set the y-axis tick labels to display without decimal precision
ax3.tick_params('y', labelsize='medium')
ax3.set_ylim(-40, 0)

# Invert the tick labels on the second y-axis such that they are displayed positive
yticks = ax3.get_yticks()
ax3.set_yticks(yticks)
#make yticks label positive and without decimals
ax3.set_yticklabels([f'{abs(y):.0f}' for y in yticks])
ax3.set_ylabel('Precipitation [mm/h]')




#resdiuals over time with precipitation
fig, ax =plt.subplots(1,1,figsize=cm2inch((15,7)))
ax.plot(dates[1:],error_preds, label='Prediction', color='darkorange', linestyle='None', marker='.', ms=msize)
ax.plot(dates[1:],error_prevobs, label='Previous observation',color='green', linestyle='None', marker='.', ms=msize)
# ax.legend(loc='lower left', fontsize='medium')
ax.set_ylabel('Residual [m]', fontsize='large')
ax.set_xlabel('Date', fontsize='large')

#adjust xticks
locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.tick_params('x', labelsize='medium')

ax1=ax.twinx()
ax1.bar(dates[1:], -X_test_rain_array[1:, 0], width=0.05, color='lightsteelblue',edgecolor='black') #alternative: cornflowerblue
# Set the y-axis tick labels to display without decimal precision
ax1.tick_params('y', labelsize='medium')
ax1.set_ylim(-40, 0)

# Invert the tick labels on the second y-axis such that they are displayed positive
yticks = ax1.get_yticks()
ax1.set_yticks(yticks)
#make yticks label positive and without decimals
ax1.set_yticklabels([f'{abs(y):.0f}' for y in yticks])
ax1.set_ylabel('Precipitation [mm/h]', rotation=-90, labelpad=11, fontsize='large')
fig.legend(loc='upper center', ncol=2, fontsize='medium', frameon=False, markerscale=5)
plt.subplots_adjust(left= 0.02, bottom=0.02,right=1.0, top=0.9, hspace=0.2)
#adjust space tight layout is taking in windows canva, neede that legend on top and label in bottom are shown. 
plt.tight_layout(rect=[0.02, 0.02 ,1.0, 0.9],pad=0.3) #rect: [left, bottom, right, top]
plt.savefig(os.path.join(respath, 'LSTM_TH1_IH1_error.png'), dpi=600)



#residuals against precipitation
plt.figure(figsize=cm2inch((15,7)))
plt.scatter(X_test_rain_array[1:], error_preds, s=msize, c='darkorange',label='Prediction')
plt.scatter(X_test_rain_array[1:], error_prevobs, s=msize, c='green',label='Previous observation')
plt.legend(fontsize='medium', markerscale=3)
plt.xlabel('Precipitation [mm/h]', fontsize='large')
plt.ylabel('Residual [m]', fontsize='large')
plt.xticks(fontsize='medium')
plt.yticks(fontsize='medium')
plt.tight_layout()
plt.savefig(os.path.join(respath, 'LSTM_TH1_IH1_res_scatter.png'), dpi=600)