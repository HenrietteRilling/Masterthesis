# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:47:15 2023

@author: Henriette
"""

import os
import glob
import re
import numpy as np
import pandas as pd

import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
#Set the number of CPU threads that PyTorch will use for parallel processing
torch.set_num_threads(8)

from datetime import datetime, timedelta
from utils import scale_data, timeseries_dataset_from_array, get_dataloader, get_WL_data, get_prcp_data, get_test_data
from plot_utils import cm2inch
from model import sampleFFNN_AR
from nn_models import LSTM_AR

from metrics import rmse, PI



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

#scale and normalise such that all data has a range between [0,1], store scaler for rescaling
X_train_sc, train_sc = scale_data(X_train)
X_val_sc = train_sc.transform(X_val)
X_test_sc = train_sc.transform(X_test)

#get scaler only for waterlevel for unscaling predictions
_, train_WL_sc=scale_data(X_train[[test_id]])
##testing
respath=r'./Results/lstm_W_10_B_1000_N_25_L_2'

#Model hyperparameter
input_size=2  #number of input features 
hidden_size=25 #number of neurons in hidden layer
output_size=1
horizon=1
batch_size=5000
window_size=10

#Initialize model
# model=sampleFFNN_AR(input_size, hidden_size, output_size)
model=LSTM_AR(input_size, window_size, hidden_size, 2)
#get paths of all weight files in the result folder using glob 
best_weights=glob.glob(os.path.join(respath, "*.pth"))
#define regular expression pattern for extracting training horizon of weight file name
pattern=re.compile(r'weights_(\d+)_\d+\.pth')
#sort weights from shortest to longest training horizon
best_weights=sorted(best_weights, key=lambda x: int(re.search(pattern, x).group(1)))

#imputation lengths the model results are plotted for
test_h=[24, 168, 672] #1h, 1d, 2d, 1 week, 1 month
# test_h=[1, 12, 24, 48, 168,  672] #1h, 1d, 2d, 1 week, 1 month

#create test features and label, as only used for plotting, use longest test horizon for creating dataset as predictions of all smaller testhorizons are inlcuded
features_test, labels_test =timeseries_dataset_from_array(X_test_sc, window_size, 168, AR=True)
dataset_test, data_loader_test=get_dataloader(features_test, labels_test, batch_size=batch_size, shuffle=False) #shuffle =False, as it is the set for validation???


#TO DO create 2 figures (?) for making another figure fo the other imputation horizon
#create figure for plots of testhorizon
fig, axes = plt.subplots(3,2, figsize=cm2inch((15, 12)), sharey=True)
msize=1
axs=axes.flatten()
i=0

import pdb
#loop over weights
all_test_pred_unsc_list=[]
for path in best_weights:
    
    #extract horizon the model was trained on using the pattern defined above, group = capturing group (\d) that was defined in pattern expression
    train_h=re.search(pattern, path).group(1)
    
    #load weights of best model
    model.load_state_dict(torch.load(path))
    model.eval()
    
    #generate predictions based on test set
    all_test_preds=[]
    for step, (x_batch_test, y_batch_test) in enumerate(data_loader_test):
        #generate prediction
        pred=model(x_batch_test)
        #save predictions of current batch
        all_test_preds.append(pred)
    
    #concat list elements along "time axis" 0
    preds_test=torch.cat(all_test_preds, 0).detach().numpy()
    #unscale all predicitons and labels
    preds_test_unsc=train_WL_sc.inverse_transform(preds_test[:,:,0])
    labels_test_unsc=train_WL_sc.inverse_transform(labels_test[:, :,0]) #include only water level
    features_test_unsc=train_WL_sc.inverse_transform(features_test[:,:,0]) #needed for PI, inlcude only water level
    all_test_pred_unsc_list.append(preds_test_unsc)


for th in [48, 168]:
    
    fig, axes = plt.subplots(3,2, figsize=cm2inch((15, 12)), sharey=True)
    msize=1
    axs=axes.flatten()
    for i, preds in enumerate(all_test_pred_unsc_list):
        if th ==48:
            #Zoom for month September:
            dates=pd.to_datetime(X_test.index)
            start_date='2022-09-03 00:00:00' #09-06
            end_date='2022-09-13 00:00:00' #09-14
            date_mask=(dates>=start_date) & (dates<end_date)
            #convert start dat to datetime string to do more operations on it later
            dt_start_date=datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            
            #if TOP: start_date
            #TOP1=np.concatenate(([np.nan],preds_test_unsc[np.argmax(date_mask), :th], np.full(np.count_nonzero(date_mask)-th-1, np.nan)))
            TOP1idx=np.where(dates=='2022-09-04 00:00:00')[0][0]
            beforeTOP1=TOP1idx-np.argmax(date_mask)
            TOP1=np.concatenate((np.full(beforeTOP1+1,np.nan),preds[TOP1idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP1-1),np.nan)))
                    
            #TOP: '2022-09-11 07:00:00' '2022-09-10 08:00:00'
            # TOP2idx=np.where(dates==(dt_start_date+timedelta(days=7)))[0][0]
            TOP2idx=np.where(dates=='2022-09-07 00:00:00')[0][0]
            beforeTOP2=TOP2idx-np.argmax(date_mask)
            TOP2=np.concatenate((np.full(beforeTOP2+1,np.nan),preds[TOP2idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP2-1),np.nan)))
            
            # TOP: '2022-09-10 00:00:00' 
            TOP3idx=np.where(dates==(dt_start_date+timedelta(days=13)))[0][0]
            TOP3idx=np.where(dates=='2022-09-08 21:00:00')[0][0]
            beforeTOP3=TOP3idx-np.argmax(date_mask)
            TOP3=np.concatenate((np.full(beforeTOP3+1,np.nan),preds[TOP3idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP3-1),np.nan)))
            
            # #TOP '2022-09-08 12:00:00'
            TOP4idx=np.where(dates=='2022-09-10 07:00:00')[0][0]
            beforeTOP4=TOP4idx-np.argmax(date_mask)
            TOP4=np.concatenate((np.full(beforeTOP4+1,np.nan),preds[TOP4idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP4-1),np.nan)))
        if th ==168:
            #Zoom for month September:
            dates=pd.to_datetime(X_test.index)
            start_date='2022-03-31 00:00:00' #09-06
            end_date='2022-05-02 00:00:00' #09-14
            date_mask=(dates>=start_date) & (dates<end_date)
            #convert start dat to datetime string to do more operations on it later
            dt_start_date=datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            
            #if TOP: start_date
            #TOP1=np.concatenate(([np.nan],preds_test_unsc[np.argmax(date_mask), :th], np.full(np.count_nonzero(date_mask)-th-1, np.nan)))
            TOP1idx=np.where(dates=='2022-04-01 00:00:00')[0][0]
            beforeTOP1=TOP1idx-np.argmax(date_mask)
            TOP1=np.concatenate((np.full(beforeTOP1+1,np.nan),preds[TOP1idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP1-1),np.nan)))
                    
            #TOP: '2022-09-11 07:00:00' '2022-09-10 08:00:00'
            # TOP2idx=np.where(dates==(dt_start_date+timedelta(days=7)))[0][0]
            TOP2idx=np.where(dates=='2022-04-08 00:00:00')[0][0]
            beforeTOP2=TOP2idx-np.argmax(date_mask)
            TOP2=np.concatenate((np.full(beforeTOP2+1,np.nan),preds[TOP2idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP2-1),np.nan)))
            
            # TOP: '2022-09-10 00:00:00' 
            # TOP3idx=np.where(dates==(dt_start_date+timedelta(days=13)))[0][0]
            TOP3idx=np.where(dates=='2022-04-15 00:00:00')[0][0]
            beforeTOP3=TOP3idx-np.argmax(date_mask)
            TOP3=np.concatenate((np.full(beforeTOP3+1,np.nan),preds[TOP3idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP3-1),np.nan)))
            
            # #TOP '2022-09-08 12:00:00'
            TOP4idx=np.where(dates=='2022-04-24 00:00:00')[0][0]
            beforeTOP4=TOP4idx-np.argmax(date_mask)
            TOP4=np.concatenate((np.full(beforeTOP4+1,np.nan),preds[TOP4idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP4-1),np.nan)))
            
            # #TOP '2022-09-08 12:00:00'
            TOP5idx=np.where(dates=='2022-04-04 11:00:00')[0][0]
            beforeTOP5=TOP5idx-np.argmax(date_mask)
            TOP5=np.concatenate((np.full(beforeTOP5+1,np.nan),preds[TOP5idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP5-1),np.nan)))
            
        ax1=axs[i]
        if i==0: #define label
            # Plot water level on the bottom axis
            ax1.plot(dates[date_mask], X_test[test_id][date_mask], color='blue', label='Observation', linestyle='None', marker='.', ms=msize)
            ax1.plot(dates[date_mask], TOP1, color='darkorange', label='Prediction', linestyle='solid', lw=0.5, marker='.', ms=msize)#alternative: limegreen, mediumseagreen
        else:#don't define label anymore
            ax1.plot(dates[date_mask], X_test[test_id][date_mask], color='blue', linestyle='None', marker='.', ms=msize)
            ax1.plot(dates[date_mask], TOP1, color='darkorange', linestyle='solid', lw=0.5, marker='.', ms=msize)#alternative: limegreen, mediumseagreen
        ax1.plot(dates[date_mask], TOP2, color='darkorange', linestyle='solid', lw=0.5, marker='.', ms=msize)
        ax1.plot(dates[date_mask], TOP3, color='darkorange', linestyle='solid', lw=0.5, marker='.', ms=msize)
        ax1.plot(dates[date_mask], TOP4, color='darkorange', linestyle='solid', lw=0.5, marker='.', ms=msize)
        if th == 168:
            ax1.plot(dates[date_mask], TOP5, color='darkorange', linestyle='solid', lw=0.5, marker='.', ms=msize)
    
        #plot lines marking TOP
        if i==0: #define lable
            ax1.axvline(x=dates[TOP1idx], color='black', linestyle='dotted',lw=1, label="TOP")
        else:
            ax1.axvline(x=dates[TOP1idx], color='black', linestyle='dotted', lw=1)
        ax1.axvline(x=dates[TOP2idx], color='black', linestyle='dotted', lw=1)
        ax1.axvline(x=dates[TOP3idx], color='black', linestyle='dotted', lw=1)
        ax1.axvline(x=dates[TOP4idx], color='black', linestyle='dotted', lw=1)
        if th ==168:
            ax1.axvline(x=dates[TOP5idx], color='black', linestyle='dotted', lw=1)
        
        #adjust xticks
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        
        #remove xlables on subplots that are not in the bottom line
        if i in [0,1,2]:
            ax1.set_xticklabels([])
            
        #set fontzise of yaxis
        ax1.tick_params('y', labelsize='medium')
           
        # Create a second y-axis for precipitation
        ax2 = ax1.twinx()
        ax2.bar(dates[date_mask], -X_test[test_prcp][date_mask], width=0.05, color='lightsteelblue',edgecolor='black') #alternative: cornflowerblue
        # Set the y-axis tick labels to display without decimal precision
        ax2.tick_params('y', labelsize='medium')
        ax2.set_ylim(-20, 0)
        
        # Invert the tick labels on the second y-axis such that they are displayed positive
        yticks = ax2.get_yticks()
        ax2.set_yticks(yticks)
        #make yticks label positive and without decimals
        ax2.set_yticklabels([f'{abs(y):.0f}' for y in yticks])
        if i in [0,2,4]: #only show y-tick lables of right column
            ax2.set_yticklabels([])
    
        ax1.set_title(f'TH-{train_h}', loc='left', fontsize='medium')
        i+=1



    fig.legend(loc='upper center', ncol=1, fontsize='medium', frameon=True, fancybox=False, edgecolor='black', bbox_to_anchor=(0.66, 0.36))  
    fig.text(0.02, 0.5, 'Water level [m]', va='center', rotation='vertical', fontsize='large')
    fig.text(0.96, 0.5, 'Precipitation [mm/h]', va='center',rotation=-90, fontsize='large')
    fig.supxlabel('Date')
    # fig.legend(loc='upper center', ncol=3, fontsize='medium', frameon=False)
    #remove subplot that isnot needed
    fig.delaxes(axes[2,1])
    
    plt.subplots_adjust(left= 0.06, bottom=0,right=0.96, top=1.0, hspace=0.2)
    #adjust space tight layout is taking in windows canva, neede that legend on top and label in bottom are shown. 
    plt.tight_layout(rect=[0.06, 0 ,0.96, 1.0],pad=0.3) #rect: [left, bottom, right, top]
    # plt.savefig(os.path.join(respath, f'Imputation_test_h_{th}_2.png'), dpi=600)
    # plt.close()

        
   


old=False
if old:    
    for th in [168]:
    # =============================================================================
    # 1. Plot for Rain event September
    # =============================================================================
    ##############
        if th ==48:
            #Zoom for month September:
            dates=pd.to_datetime(X_test.index)
            start_date='2022-09-03 00:00:00' #09-06
            end_date='2022-09-13 00:00:00' #09-14
            date_mask=(dates>=start_date) & (dates<end_date)
            #convert start dat to datetime string to do more operations on it later
            dt_start_date=datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            
            #if TOP: start_date
            #TOP1=np.concatenate(([np.nan],preds_test_unsc[np.argmax(date_mask), :th], np.full(np.count_nonzero(date_mask)-th-1, np.nan)))
            TOP1idx=np.where(dates=='2022-09-04 00:00:00')[0][0]
            beforeTOP1=TOP1idx-np.argmax(date_mask)
            TOP1=np.concatenate((np.full(beforeTOP1+1,np.nan),preds_test_unsc[TOP1idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP1-1),np.nan)))
                    
            #TOP: '2022-09-11 07:00:00' '2022-09-10 08:00:00'
            # TOP2idx=np.where(dates==(dt_start_date+timedelta(days=7)))[0][0]
            TOP2idx=np.where(dates=='2022-09-07 00:00:00')[0][0]
            beforeTOP2=TOP2idx-np.argmax(date_mask)
            TOP2=np.concatenate((np.full(beforeTOP2+1,np.nan),preds_test_unsc[TOP2idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP2-1),np.nan)))
            
            # TOP: '2022-09-10 00:00:00' 
            TOP3idx=np.where(dates==(dt_start_date+timedelta(days=13)))[0][0]
            TOP3idx=np.where(dates=='2022-09-08 21:00:00')[0][0]
            beforeTOP3=TOP3idx-np.argmax(date_mask)
            TOP3=np.concatenate((np.full(beforeTOP3+1,np.nan),preds_test_unsc[TOP3idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP3-1),np.nan)))
            
            # #TOP '2022-09-08 12:00:00'
            TOP4idx=np.where(dates=='2022-09-10 07:00:00')[0][0]
            beforeTOP4=TOP4idx-np.argmax(date_mask)
            TOP4=np.concatenate((np.full(beforeTOP4+1,np.nan),preds_test_unsc[TOP4idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP4-1),np.nan)))
        if th ==168:
            #Zoom for month September:
            dates=pd.to_datetime(X_test.index)
            start_date='2022-03-31 00:00:00' #09-06
            end_date='2022-05-02 00:00:00' #09-14
            date_mask=(dates>=start_date) & (dates<end_date)
            #convert start dat to datetime string to do more operations on it later
            dt_start_date=datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            
            #if TOP: start_date
            #TOP1=np.concatenate(([np.nan],preds_test_unsc[np.argmax(date_mask), :th], np.full(np.count_nonzero(date_mask)-th-1, np.nan)))
            TOP1idx=np.where(dates=='2022-04-01 00:00:00')[0][0]
            beforeTOP1=TOP1idx-np.argmax(date_mask)
            TOP1=np.concatenate((np.full(beforeTOP1+1,np.nan),preds_test_unsc[TOP1idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP1-1),np.nan)))
                    
            #TOP: '2022-09-11 07:00:00' '2022-09-10 08:00:00'
            # TOP2idx=np.where(dates==(dt_start_date+timedelta(days=7)))[0][0]
            TOP2idx=np.where(dates=='2022-04-08 00:00:00')[0][0]
            beforeTOP2=TOP2idx-np.argmax(date_mask)
            TOP2=np.concatenate((np.full(beforeTOP2+1,np.nan),preds_test_unsc[TOP2idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP2-1),np.nan)))
            
            # TOP: '2022-09-10 00:00:00' 
            # TOP3idx=np.where(dates==(dt_start_date+timedelta(days=13)))[0][0]
            TOP3idx=np.where(dates=='2022-04-15 00:00:00')[0][0]
            beforeTOP3=TOP3idx-np.argmax(date_mask)
            TOP3=np.concatenate((np.full(beforeTOP3+1,np.nan),preds_test_unsc[TOP3idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP3-1),np.nan)))
            
            # #TOP '2022-09-08 12:00:00'
            TOP4idx=np.where(dates=='2022-04-24 00:00:00')[0][0]
            beforeTOP4=TOP4idx-np.argmax(date_mask)
            TOP4=np.concatenate((np.full(beforeTOP4+1,np.nan),preds_test_unsc[TOP4idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP4-1),np.nan)))
            
            # #TOP '2022-09-08 12:00:00'
            TOP5idx=np.where(dates=='2022-04-04 11:00:00')[0][0]
            beforeTOP5=TOP5idx-np.argmax(date_mask)
            TOP5=np.concatenate((np.full(beforeTOP5+1,np.nan),preds_test_unsc[TOP5idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP5-1),np.nan)))
            
        ax1=axs[i]
        if i==0: #define label
            # Plot water level on the bottom axis
            ax1.plot(dates[date_mask], X_test[test_id][date_mask], color='blue', label='Observation', linestyle='None', marker='.', ms=msize)
            ax1.plot(dates[date_mask], TOP1, color='darkorange', label='Prediction', linestyle='solid', lw=0.5, marker='.', ms=msize)#alternative: limegreen, mediumseagreen
        else:#don't define label anymore
            ax1.plot(dates[date_mask], X_test[test_id][date_mask], color='blue', linestyle='None', marker='.', ms=msize)
            ax1.plot(dates[date_mask], TOP1, color='darkorange', linestyle='solid', lw=0.5, marker='.', ms=msize)#alternative: limegreen, mediumseagreen
        ax1.plot(dates[date_mask], TOP2, color='darkorange', linestyle='solid', lw=0.5, marker='.', ms=msize)
        ax1.plot(dates[date_mask], TOP3, color='darkorange', linestyle='solid', lw=0.5, marker='.', ms=msize)
        ax1.plot(dates[date_mask], TOP4, color='darkorange', linestyle='solid', lw=0.5, marker='.', ms=msize)
        if th == 168:
            ax1.plot(dates[date_mask], TOP5, color='darkorange', linestyle='solid', lw=0.5, marker='.', ms=msize)
    
        #plot lines marking TOP
        if i==0: #define lable
            ax1.axvline(x=dates[TOP1idx], color='black', linestyle='dotted',lw=1, label="TOP")
        else:
            ax1.axvline(x=dates[TOP1idx], color='black', linestyle='dotted', lw=1)
        ax1.axvline(x=dates[TOP2idx], color='black', linestyle='dotted', lw=1)
        ax1.axvline(x=dates[TOP3idx], color='black', linestyle='dotted', lw=1)
        ax1.axvline(x=dates[TOP4idx], color='black', linestyle='dotted', lw=1)
        if th ==168:
            ax1.axvline(x=dates[TOP5idx], color='black', linestyle='dotted', lw=1)
        
        #adjust xticks
        import pdb
        pdb.set_trace()
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        
        #remove xlables on subplots that are not in the bottom line
        if i in [0,1,2]:
            ax1.set_xticklabels([])
            
        #set fontzise of yaxis
        ax1.tick_params('y', labelsize='medium')
           
        # Create a second y-axis for precipitation
        ax2 = ax1.twinx()
        ax2.bar(dates[date_mask], -X_test[test_prcp][date_mask], width=0.05, color='lightsteelblue',edgecolor='black') #alternative: cornflowerblue
        # Set the y-axis tick labels to display without decimal precision
        ax2.tick_params('y', labelsize='medium')
        ax2.set_ylim(-20, 0)
        
        # Invert the tick labels on the second y-axis such that they are displayed positive
        yticks = ax2.get_yticks()
        ax2.set_yticks(yticks)
        #make yticks label positive and without decimals
        ax2.set_yticklabels([f'{abs(y):.0f}' for y in yticks])
        if i in [0,2,4]: #only show y-tick lables of right column
            ax2.set_yticklabels([])
    
        ax1.set_title(f'TH-{train_h}', loc='left', fontsize='medium')
        i+=1



# fig.legend(loc='upper center', ncol=1, fontsize='medium', frameon=True, fancybox=False, edgecolor='black', bbox_to_anchor=(0.64, 0.36))  
# fig.text(0.02, 0.5, 'Water level [m]', va='center', rotation='vertical', fontsize='large')
# fig.text(0.96, 0.5, 'Precipitation [mm/h]', va='center',rotation=-90, fontsize='large')
# fig.supxlabel('Date')
# # fig.legend(loc='upper center', ncol=3, fontsize='medium', frameon=False)
# #remove subplot that isnot needed
# fig.delaxes(axes[2,1])

# plt.subplots_adjust(left= 0.06, bottom=0,right=0.96, top=1.0, hspace=0.2)
# #adjust space tight layout is taking in windows canva, neede that legend on top and label in bottom are shown. 
# plt.tight_layout(rect=[0.06, 0 ,0.96, 1.0],pad=0.3) #rect: [left, bottom, right, top]
# plt.savefig(os.path.join(respath, f'Imputation_test_h_{th}_2.png'), dpi=600)
# plt.close()
