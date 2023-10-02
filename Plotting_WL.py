# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 08:20:43 2023

@author: Henriette
"""

#import packages
import os
import glob 
import csv
import re #package for "regular expressions", used for searching, matching and manipulating text
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, DatetimeTickFormatter

from Data_loader import get_WL_data



def plot_water_level_subplot(df, rows, cols, id_dict, savepath=None):
    
    fig, axes=plt.subplots(rows, cols, figsize=(12,8))
    
    for i, column in enumerate(df.columns):
        row=i//cols
        col=i%cols
        ax=axes[row, col]
        ax.plot(df[column])
        ax.set_title(f'{id_dict.get(column)}')
        # Set shared x and y labels
    fig.text(0.5, 0, 'Date', ha='center', fontsize=14)
    fig.text(0, 0.5, 'Water level [m]', va='center', rotation='vertical', fontsize=14)
#        ax.set_xlabel('Date'); ax.set_ylabel('Water level [m]')
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=300)
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
    
    rows=7
    cols=3
    # plot_water_level_subplot(WL, rows, cols, station_id_to_name)
    
    # for col in WL.columns:
    #     plot_water_level(WL[[col]], station_id_to_name.get(col))
        
for col in WL.columns:
    plot_water_level_bokeh(WL[[col]],  station_id_to_name.get(col))        
