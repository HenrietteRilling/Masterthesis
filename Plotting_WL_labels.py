# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:22:52 2023

@author: Henriette
"""

import os
import pandas as pd
import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



def plot_with_labels(df, stat_id, name):
    
    #make sure that only period, in which data is available is plotted
    df=df[(df.index>df[stat_id].first_valid_index())&(df.index<df[stat_id].last_valid_index())] 
    
    #get labels
    lbls=np.sort(df[f'{stat_id}_label'].unique())
    
    label_mapping = {0: 'Normal Signal', 1:'Sensor Error' , 2:'Missing Values', 3:'Outlier'}
    colors = ['white','cyan' ,'blue', 'red']  # Replace with your preferred colors

    # Create a custom colormap
    custom_cmap = mcolors.ListedColormap(colors)  # Use only the necessary colors
    
    plt.figure()
    # Your code to create the plot and background color
    ax = df[f'{stat_id}'].plot(color='blue')  # Set the value data color to blue
    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  df[f'{stat_id}_label'].values[np.newaxis],
                  cmap=custom_cmap, alpha=0.2)  # Adjust alpha for lighter hues

    # Create legend handles with custom labels
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(colors))]
    #custom_legend_labels = [label_mapping[label] for label in lbls]
    custom_legend_labels = label_mapping.values()
    # Create the legend with custom labels
    legend=plt.legend(legend_handles, custom_legend_labels, title='Labels', fontsize= 'small',alignment='left')
    # Adjust the legend's background color to be less transparent
    legend.get_frame().set_alpha(1.0)  # Set alpha to 1 for fully opaque background

    plt.title(f'{name}')
    plt.xlabel('Date'); plt.ylabel('Water level [m]')

    # plt.savefig(os.path.join(os.path.relpath(r'.\Label_plots')+f'\{stat_id}_labeled.png'), dpi=300)
    plt.show()


def plot_with_labels_subplots(WL, labels, id_dict,rows, cols):
    fig, axes=plt.subplots(rows, cols, figsize=(16,10))
    
    for i, col in enumerate(WL.columns):
        #merge WL data and labels for one station
        df=WL[[col]].join(labels[[col]], rsuffix='_label')
        
        #make sure that only period, in which data is available is plotted
        df=df[(df.index>df[col].first_valid_index())&(df.index<df[col].last_valid_index())] 
        
        #get labels
        lbls=np.sort(df[f'{col}_label'].unique())
        
        label_mapping = {0: 'Normal Signal', 1:'Sensor Error' , 2:'Missing Values', 3:'Outlier'}
        colors = ['white','cyan' ,'blue', 'red']  # Replace with your preferred colors
    
        # Create a custom colormap
        custom_cmap = mcolors.ListedColormap(colors)  # Use only the necessary colors
        
        row=i//cols
        column=i%cols
        ax=axes[row, column]
        ax.plot(df[f'{col}'], color='blue')  # Set the value data color to blue
        ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                      df[f'{col}_label'].values[np.newaxis],
                      cmap=custom_cmap, alpha=0.2)  # Adjust alpha for lighter hues
        ax.set_title(f'{id_dict.get(col)}')
    
    # Create legend handles with custom labels
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i]) for i in range(len(colors))]
    #custom_legend_labels = [label_mapping[label] for label in lbls]
    custom_legend_labels = label_mapping.values()
    # Create the legend with custom labels
    legend=axes[0,0].legend(legend_handles, custom_legend_labels, title='Labels', fontsize= 'small',alignment='left')
    # Adjust the legend's background color to be less transparent
    #legend.get_frame().set_alpha(1.0)  # Set alpha to 1 for fully opaque background

    # Set shared x and y labels
    
    fig.text(0.5, 0, 'Date', ha='center', fontsize=14)
    fig.text(0, 0.5, 'Water level m]', va='center', rotation='vertical', fontsize=14)  
    
    plt.tight_layout()
#    plt.savefig(os.path.join(os.path.relpath(r'.\Label_plots')+f'\ll_stations_labeled.png'), dpi=300)
    plt.show()

def plot_simple_subplots(df, id_dict, rows, cols):
    fig, axes=plt.subplots(rows, cols, figsize=(16,10)) 
    
    for i, col in enumerate(df.columns):
        row=i//cols
        column=i%cols
        ax=axes[row, column]        
        ax.plot(df[col])
        ax.set_title(f'{id_dict.get(col)}')
        ax.axhline(y = 3, color = 'r', linestyle = '-')
        ax.axhline(y = -3, color = 'r', linestyle = '-')

    # Set shared x and y labels
    fig.text(0.5, 0, 'Date', ha='center', fontsize=14)
    fig.text(0, 0.5, 'Z-score', va='center', rotation='vertical', fontsize=14)
        
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.relpath(r'.\Label_plots')+f'\zscore.png'), dpi=300)
    plt.show()    

if __name__ == "__main__":
    #get datapath of folder with data
    datafolder=r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\Data_WL'
    
    #load WL data with labels
    data=pd.read_csv(os.path.join(datafolder, 'WL_w_labels.csv'), sep=',')
    #convert date column to pd.datetime string
    data['date']=pd.to_datetime(data.iloc[:,0],format='%Y-%m-%d %H:%M:%S')
    #set date column to index
    data.set_index('date', inplace=True)
    
    #load dictionaries with information on station names
    with open(os.path.join(datafolder, 'station_name_to_id.pkl'), 'rb') as file:
        station_name_to_id=pickle.load(file)
    
    with open(os.path.join(datafolder, 'station_id_to_name.pkl'), 'rb') as file:
        station_id_to_name=pickle.load(file)
    
    
    #dataframe contains both the WL data as well as the labels, let's split the data in 2 separate dataframes
    #WL data is in the first 21 columns, lables in the remaining 21 columns
    WL=data.iloc[:, :len(station_name_to_id)].copy()
    labels=data.iloc[:, len(station_name_to_id):].copy()
    #rename label columns as their name changed while loading
    labels.columns=WL.columns
    

    #__________Plot one station_______________________   
    # test_id=station_name_to_id.get('Åbro')
    # plot_with_labels(WL[[test_id]].join(labels[[test_id]], rsuffix='_label'), test_id, station_id_to_name.get(test_id))
    
    #__________Plot all stations_______________________
    # for station_id, station_name in station_id_to_name.items():
    # #create one dataframe with station data + lables for plotting
    #     df=WL[[station_id]].join(labels[[station_id]], rsuffix='_label') 
    #     plot_with_labels(df, station_id, station_name)
    
    #_________Plot all stations in one big plot__________    
    plot_with_labels_subplots(WL, labels, station_id_to_name, rows=7, cols=3)
    
    #______________Plot Z_scores______________________
    # z_scores=(WL-WL.mean())/WL.std()
    # plot_simple_subplots(z_scores, station_id_to_name, rows=7, cols=3)

    # for col in z_scores:
    #     plt.figure()
    #     plt.plot(z_scores[col])
    #     plt.axhline(y = 3, color = 'r', linestyle = '-')
    #     plt.axhline(y = -3, color = 'r', linestyle = '-')
    #     plt.title(station_id_to_name.get(col))


