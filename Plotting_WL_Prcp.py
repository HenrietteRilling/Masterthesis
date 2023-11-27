# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 08:20:43 2023

@author: Henriette
"""

#import packages

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, DatetimeTickFormatter

from Data_loader import get_WL_data, get_prcp_data, get_test_data
from plot_utils import cm2inch



def plot_water_level_subplot(df, rows, cols, id_dict, savepath=None):
    
    fig, axes=plt.subplots(rows, cols, figsize=(cm2inch(15, 20)), layout='constrained')
    
    for i, column in enumerate(df.columns):
        row=i//cols
        col=i%cols
        ax=axes[row, col]
        ax.plot(df[column], color='blue')
        ax.set_title(f'{id_dict.get(column)}', fontsize='small')
        ax.tick_params(axis='x', labelsize='small')
        ax.tick_params(axis='y', labelsize='small')
        
        #Format x-axis ticks such that only some dates are shown
        locator = mdates.AutoDateLocator(minticks=3, maxticks=4)
        ax.xaxis.set_major_locator(locator)
        
        # Format x-axis ticks  of Højlund and Møllerup to show only years and not months
        if id_dict.get(column) in ('Møllerup', 'Højlund'):
            #set formatter to years
            date_fmt = mdates.DateFormatter('%Y')
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(date_fmt)
            ax.tick_params(axis='x', labelsize='small')
            
        #set global x-axis label
        if id_dict.get(column) == 'Langå':
            ax.set_xlabel('Date', fontsize='medium')

        #set global y-axis label
        if id_dict.get(column) == 'Højlund':
            ax.set_ylabel('Water level [m]', fontsize='medium')
    
    # Set shared x and y labels
    # fig.text(0.5, 0, 'Date', ha='center', fontsize=14)
    # fig.text(0, 0.5, 'Water level [m]', va='center', rotation='vertical', fontsize=14)
    

 # Set xtick labels fontsize to 34

    # Set ytick labels fontsize to 34
    # plt.yticks(fontsize='medium')
    #adjust the size of the figure window
    # plt.gcf().set_size_inches(cm2inch(50,30))
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

def plot_prcp(df, rows, cols, savepath=None):
    fig, axes=plt.subplots(rows, cols, figsize=(cm2inch(15, 12)), constrained_layout=True)
    legend_colors = ['cornflowerblue', 'royalblue']
    for i, column in enumerate(prcp.columns):
        row=i//cols
        col=i%cols
        ax=axes[row, col]
        stat_data=get_test_data(column, df)
        if column in prcp.columns[:7]:
            color= legend_colors[0]
        else: color = legend_colors[1]
        
        ax.plot(stat_data, color=color)
        ax.set_title(f'{column}', fontsize='small')
        ax.tick_params(axis='x', labelsize='small')
        ax.tick_params(axis='y', labelsize='small')
        
        locator = mdates.AutoDateLocator(minticks=3, maxticks=4)
        # formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        # ax.xaxis.set_major_formatter(formatter)


        # Format x-axis ticks  of Højlund and Møllerup to show only years and not months
        if column in ('5124', '5162'):
            #set formatter to years
            date_fmt = mdates.DateFormatter('%Y')
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(date_fmt)
            ax.tick_params(axis='x', labelsize='small')        
            
        #set global x-axis label
        if column == '5225':
            ax.set_xlabel('Date', fontsize='medium')

        #set global y-axis label
        if column == '5190':
            ax.set_ylabel('Precipitation [mm/h]', fontsize='medium')
        

    
    # Delete the subplot at row=4, col=3    
    plt.delaxes(axes[3, 2])
    #create legend explaining different colors in plot
    legend_labels = ['SVK', 'DMI']    
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(legend_colors, legend_labels)]
   
    legend_ax = fig.add_axes([0.72, 0.1, 0.1, 0.2])  # Adjust the coordinates and size as needed
    legend_ax.set_axis_off()
    legend_ax.legend(handles=legend_patches, loc='center', fontsize='medium')

    if savepath:
        plt.savefig(savepath, dpi=600)
        plt.close()
    plt.show()

        
if __name__ == "__main__":
    #load data
    WL, _, _, station_id_to_name = get_WL_data(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\Data_WL')
    
    # #remove outliers based on z-score
    # zscore=(WL-WL.mean())/WL.std()
    # threshold=3
    # WL_wo_anom= WL 
    # #Remove anomalies from ns Uldumkær
    # for col in WL.columns:
    #     WL_wo_anom[col][np.abs(zscore[col])>threshold]=np.nan

    rows=7
    cols=3
    plot_water_level_subplot(WL, rows, cols, station_id_to_name)
#     # plot_water_level(WL_wo_anom[['211711']], station_id_to_name.get('211711'))



#     '''Plotting examples'''
    # savepth=r'C:\Users\henri\Documents\Universität\Masterthesis\Report\WL_all_stations2.png'
    # plot_water_level_subplot(WL, rows, cols, station_id_to_name, savepath=savepth)

# for col in WL.columns:
#     plot_water_level(WL[[col]], station_id_to_name.get(col))

        
# for col in WL.columns:
#     plot_water_level_bokeh(WL[[col]],  station_id_to_name.get(col))        
 
   #####################Precipitation##############################
    # prcp=get_prcp_data(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\SVK', r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\DMI_Climate_Data_prcp', join=True)
    
    # prcp.columns=prcp.columns.str.lstrip('0')
    # # savepath=r'C:\Users\henri\Documents\Universität\Masterthesis\Report\Prcp_all_stations2.png'
    # plot_prcp(prcp, 4, 3)

    
