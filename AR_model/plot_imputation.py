# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:55:25 2023

@author: Henriette
"""

import os
import pickle
import csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime, timedelta
from plot_utils import cm2inch





#respath=r'C:\Users\henri\Desktop\LSTM_preliminary' # old results
# respath=r'C:\Users\henri\Documents\Universität\Masterthesis\Results\LSTM_wo_prcp'
#respath=r'C:\Users\henri\Documents\Universität\Masterthesis\Results\LSTM_AR'
respath=r'C:\Users\henri\Documents\Universität\Masterthesis\Results\LSTM_AR_Bjerrinbro_station'
configpath=os.path.join(respath, 'configs.csv')
save=True

#Read csv file with model configurations
with open(configpath, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Convert the CSV rows to a list
    config_list = list(csv_reader)


#Constants
plot_h=[48, 168]
train_h=[1, 12, 24, 48, 168]
window=[10, 20, 50]

for th in plot_h:    
    fig, axes = plt.subplots(3,2, figsize=cm2inch((15, 12)), sharey=True)
    msize=1
    axs=axes.flatten()
    colorlist=['darkorange', 'cyan', 'lime']   
    c=-1

    for config in config_list:
        #get path of pkl file for current model configuration
        pred_pkl_path=os.path.join(respath, f'{os.path.basename(config[0])}.pkl')
        #Read pickle
        if os.path.exists(pred_pkl_path):    
            with open(pred_pkl_path, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                c+=1
        else: continue
        
        #Extract data
        pred_list=data[0]
        X_test=data[1]
        
        #Get id of test station and precipitation station
        test_id=config[-2]
        test_prcp=config[-1]
        
        
    
        for i, preds in enumerate(pred_list):
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
                end_date='2022-04-24 00:00:00' #09-14
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
                TOP4idx=np.where(dates=='2022-04-04 11:00:00')[0][0]
                beforeTOP4=TOP4idx-np.argmax(date_mask)
                TOP4=np.concatenate((np.full(beforeTOP4+1,np.nan),preds[TOP4idx,:th], np.full((np.count_nonzero(date_mask)-th-beforeTOP4-1),np.nan)))
                               
            ax1=axs[i]
            if i==0:
                if c==0: #define label
                    # Plot water level on the bottom axis
                    ax1.plot(dates[date_mask], X_test[test_id][date_mask], color='blue', label='Observation', linestyle='None', marker='.', ms=msize)
                    ax1.axvline(x=dates[TOP1idx], color='black', linestyle='dotted',lw=1, label="TOP")
                ax1.plot(dates[date_mask], TOP1, color=colorlist[c], label=f'Prediction W={window[c]} h', linestyle='solid', lw=0.5, marker='.', ms=msize)#alternative: limegreen, mediumseagreen

            else:#don't define label anymore
                ax1.plot(dates[date_mask], X_test[test_id][date_mask], color='blue', linestyle='None', marker='.', ms=msize)
                ax1.plot(dates[date_mask], TOP1, color=colorlist[c], linestyle='solid', lw=0.5, marker='.', ms=msize)#alternative: limegreen, mediumseagreen
                ax1.axvline(x=dates[TOP1idx], color='black', linestyle='dotted', lw=1)

            ax1.plot(dates[date_mask], TOP2, color=colorlist[c], linestyle='solid', lw=0.5, marker='.', ms=msize)
            ax1.plot(dates[date_mask], TOP3, color=colorlist[c], linestyle='solid', lw=0.5, marker='.', ms=msize)
            ax1.plot(dates[date_mask], TOP4, color=colorlist[c], linestyle='solid', lw=0.5, marker='.', ms=msize)
        
            ax1.axvline(x=dates[TOP2idx], color='black', linestyle='dotted', lw=1)
            ax1.axvline(x=dates[TOP3idx], color='black', linestyle='dotted', lw=1)
            ax1.axvline(x=dates[TOP4idx], color='black', linestyle='dotted', lw=1)
            
            #adjust xticks
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            ax1.xaxis.set_major_locator(locator)
            ax1.xaxis.set_major_formatter(formatter)
            if test_id =='211711':
                ax1.set_ylim(bottom=49.7, top=50.2) #TODO
            else:
                ax1.set_ylim(bottom=3, top=4) #TODO
            
            #remove xlables on subplots that are not in the bottom line
            if i in [0,1,2]:
                ax1.set_xticklabels([])
                
            #set fontzise of yaxis
            ax1.tick_params('y', labelsize='medium')
            if test_prcp == 'None':
                continue
            
            else:
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
        
            ax1.set_title(f'TH-{train_h[i]}', loc='left', fontsize='medium')

    fig.legend(loc='upper center', ncol=1, fontsize='medium', frameon=True, fancybox=False, edgecolor='black', bbox_to_anchor=(0.7, 0.36))  
    fig.text(0.02, 0.5, 'Water level [m]', va='center', rotation='vertical', fontsize='large')
    if not test_prcp =='None':
        fig.text(0.96, 0.5, 'Precipitation [mm/h]', va='center',rotation=-90, fontsize='large')
    fig.supxlabel('Date')
    # fig.legend(loc='upper center', ncol=3, fontsize='medium', frameon=False)
    #remove subplot that isnot needed
    fig.delaxes(axes[2,1])
    
    plt.subplots_adjust(left= 0.06, bottom=0,right=0.96, top=1.0, hspace=0.2)
    #adjust space tight layout is taking in windows canva, neede that legend on top and label in bottom are shown. 
    plt.tight_layout(rect=[0.06, 0 ,0.96, 1.0],pad=0.3) #rect: [left, bottom, right, top]
    if save:
        plt.savefig(os.path.join(respath, f'Imputation_h_{th}_w_windows_zoom.png'), dpi=600)
        plt.close()
    # plt.show()



        
