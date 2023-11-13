# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 08:20:43 2023

@author: Henriette
"""

#import packages

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, DatetimeTickFormatter

from Data_loader import get_WL_data
from plot_utils import cm2inch



def plot_water_level_subplot(df, rows, cols, id_dict, savepath=None):
    
    fig, axes=plt.subplots(rows, cols, figsize=(cm2inch(15, 20)), layout='constrained')
    
    for i, column in enumerate(df.columns):
        row=i//cols
        col=i%cols
        ax=axes[row, col]
        ax.plot(df[column], color='blue')
        ax.set_title(f'{id_dict.get(column)}', fontsize='xx-large')
        ax.tick_params(axis='x', labelsize='x-large')
        ax.tick_params(axis='y', labelsize='x-large')
        
        # Format x-axis ticks  of Højlund and Møllerup to show only years and not months
        if id_dict.get(column) in ('Møllerup', 'Højlund'):
            #set formatter to years
            date_fmt = mdates.DateFormatter('%Y')
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(date_fmt)
            ax.tick_params(axis='x', labelsize='x-large')
            
        #set global x-axis label
        if id_dict.get(column) == 'Langå':
            ax.set_xlabel('Date', fontsize='xx-large')

        #set global y-axis label
        if id_dict.get(column) == 'Højlund':
            ax.set_ylabel('Water level [m]', fontsize='xx-large')
    
    # Set shared x and y labels
    # fig.text(0.5, 0, 'Date', ha='center', fontsize=14)
    # fig.text(0, 0.5, 'Water level [m]', va='center', rotation='vertical', fontsize=14)
    

 # Set xtick labels fontsize to 34

    # Set ytick labels fontsize to 34
    # plt.yticks(fontsize='medium')
    #adjust the size of the figure window
    plt.gcf().set_size_inches(cm2inch(50,30))
    if savepath:
        plt.savefig(savepath, dpi=600)
        plt.close()
        
    else:
        plt.show()


def plot_water_level(df, name, savepath=None):
    plt.figure()
    plt.plot(df)
    plt.title(name)
    plt.xlabel('Date');plt.ylabel('Water level [m]');
    
    if savepath:
        plt.savefig(savepath + f'{station_id_list[i]}')
        plt.close()
    else:
        plt.show()


def plot_water_level_bokeh(df, name):
    # Create a Bokeh figure
    p = figure(title=name, x_axis_label='Date', y_axis_label='Water level [m]')
    
    # Convert the DataFrame to a ColumnDataSource for Bokeh
    source = ColumnDataSource(df)
    
    # Create a line plot using Bokeh's line function
    p.line(x='date', y=df.columns[0], source=source, line_width=2, line_color='blue')
    # Customize the x-axis date format using DatetimeTickFormatter
    p.xaxis.formatter = DatetimeTickFormatter(days='%Y-%m-%d', months='%Y-%m', years='%Y')
    show(p)
        
        
if __name__ == "__main__":
    #load data
    WL, _, _, station_id_to_name = get_WL_data(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\Data_WL')
    
    #remove outliers based on z-score
    zscore=(WL-WL.mean())/WL.std()
    threshold=3
    WL_wo_anom= WL 
    #Remove anomalies from ns Uldumkær
    for col in WL.columns:
        WL_wo_anom[col][np.abs(zscore[col])>threshold]=np.nan

    rows=7
    cols=3
#    plot_water_level_subplot(WL, rows, cols, station_id_to_name)
    # plot_water_level(WL_wo_anom[['211711']], station_id_to_name.get('211711'))



    '''Plotting examples'''
    savepth=r'C:\Users\henri\Documents\Universität\Masterthesis\Report\WL_all_stations.png'
    plot_water_level_subplot(WL, rows, cols, station_id_to_name, savepath=savepth)

# for col in WL.columns:
#     plot_water_level(WL[[col]], station_id_to_name.get(col))

        
# for col in WL.columns:
#     plot_water_level_bokeh(WL[[col]],  station_id_to_name.get(col))        
