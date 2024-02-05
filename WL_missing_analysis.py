# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:49:11 2023

@author: Henriette
"""

import os
import pandas as pd
import pickle
import numpy as np

import matplotlib.pyplot as plt
import missingno as msno

from Data_loader import get_WL_data
from utils import cm2inch


#Function to calculate length, frequency and other statistics of gaps
def get_gaps(df, id):

    # Create a list to store gap lengths
    gap_lengths = []

    # Initialize variables to track the start and end of a gap
    gap_start = None

    # Iterate through the DataFrame
    for idx, row in df.iterrows():
        if pd.isna(row[id]):
            # If water_level is missing, it's the start of a gap
            if gap_start is None:
                gap_start = idx
        else:
            # If water_level is not missing, it's the end of a gap
            if gap_start is not None:
                gap_end = idx
                gap_length = (gap_end - gap_start).total_seconds() / 3600  # Convert to hours, there is no attribute directly outputting the hours
                gap_lengths.append(gap_length)
                gap_start = None  # Reset gap_start

    # Convert gap_lengths to a DataFrame and calculate frequency and relative frequency
    gap_df = pd.DataFrame(gap_lengths, columns=['Gap Length (hours)'])
    gap_freq = gap_df['Gap Length (hours)'].value_counts().reset_index()
    gap_freq.columns = ['Gap Length (hours)', 'Frequency']
    gap_freq['Relative Frequency']= gap_freq['Frequency']/len(gap_lengths)*100
    gap_freq.sort_values(by='Gap Length (hours)', inplace=True)
    gap_freq['Cumsum']=gap_freq['Relative Frequency'].cumsum()
    return gap_freq

def plot_gaps(df, station_id ,station_name):    
    plt.figure()
    plt.plot(df['Gap Length (hours)'], df['Relative Frequency'], linestyle='', marker='.')
    plt.xscale('log')
    plt.xlabel('Gap length [h]'); plt. ylabel('Relative frequency')
    plt.title(station_name)
    plt.savefig(os.path.join(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\WL_Missing_plots'+f'\{station_id}.png'), dpi=300)
    plt.show()

def plot_statistics(df):
    fig, ax = plt.subplots(2,2,figsize=(12,8))
    df['Nr'].plot.bar(ax=ax[0,0], sharex=True, title='Total number of gaps', ylabel='Number of gaps', color='red')
    df['Most_frequent'].plot.bar(ax=ax[1,1], sharex=True, title='Most frequent', ylabel='Gap length [h]')
    df['Max'].plot.bar(ax=ax[0,1],title='Max' ,ylabel='Gap length [h]')
    df['Min'].plot.bar(ax=ax[1,0], sharex=True, title='Min', ylabel='Gap length [h]')
    plt.savefig(os.path.join(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\WL_Missing_plots'+'\statistics.png'), dpi=300)
    plt.show()
    
if __name__ == "__main__":
    #load data
    WL, _, _, station_id_to_name = get_WL_data(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\Data_WL')
    
    #Init list for storing gap_lengths
    gaps=[]
    #Create dataframe to store information on gaps    
    gap_info_df=pd.DataFrame(index=station_id_to_name.values(), columns=['Nr', 'Max', 'Min', 'Most_frequent'])
    
    
    for station_id, name in station_id_to_name.items():
        #extract data for respective station
        df=WL[[station_id]]
        #make sure that only period in which sensor data is available is used
        df=df[(df.index>df.first_valid_index())&(df.index<df.last_valid_index())] 
        #calculate gap lengts and statistics
        gap_df=get_gaps(df, station_id)
        #add to list
        gaps.append(gap_df)
       
        #Calculate some more statistics
        gap_info_df.loc[[name],['Nr']]=gap_df['Frequency'].sum()
        gap_info_df.loc[[name],['Max']]=gap_df['Gap Length (hours)'].max()
        gap_info_df.loc[[name],['Min']]=gap_df['Gap Length (hours)'].min()
        
        #Get an idea which gap length is most frequent
        gap_info_df.loc[[name],['Most_frequent']]=(gap_df.loc[gap_df[gap_df['Relative Frequency']==gap_df['Relative Frequency'].max()].index[0],['Gap Length (hours)']])[0]
        
        
        

# =============================================================================
# Plotting
# =============================================================================

# #Statistics
# plot_statistics(gap_info_df)

# #one plot per station    
# for i, key in enumerate(station_id_to_name.keys()):
#     plot_gaps(gaps[i], key ,station_id_to_name.get(key))


#subplots with all stations
rows=7
cols=3
fig, axes=plt.subplots(rows, cols, figsize=(12,8), sharex=True) 

for i, station in enumerate(station_id_to_name.values()):
    df=gaps[i]
    #get row and column for current subplot
    row=i//cols
    column=i%cols
    ax=axes[row, column]

    ax.plot(df['Gap Length (hours)'], df['Cumsum'], linestyle='', marker='.')
    ax.set_xscale('log')
    ax.set_title(station)

# Set shared x and y labels
fig.text(0.5, 0, 'Gap length [h]', ha='center', fontsize=14)
fig.text(0, 0.5, 'Relative frequency', va='center', rotation='vertical', fontsize=14)

plt.tight_layout()
# plt.savefig(os.path.join(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\WL_Missing_plots\subplots.png'), dpi=300)
# #plt.close()    

plt.figure(figsize=cm2inch((9, 6.5)))
for i, station in enumerate(station_id_to_name.values()):

    df=gaps[i]
    plt.plot(df['Gap Length (hours)'], df['Cumsum'], linestyle='', marker='.', ms=3)
    plt.xscale('log')

plt.xlabel('Gap length [h]', fontsize='large')
plt.ylabel('Relative cumulative\nfrequency [%]', fontsize='large')
plt.tick_params(labelsize='medium')
plt.xlim(0,10**5)
plt.hlines(90, 0, 10**5, colors='black', linestyles='dashed')
plt.tight_layout()
plt.savefig(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\WL_Missing_plots\cum_freq_all_zoom.png', dpi=600)


#make one dataframe with all observations to get overall cumulative frequency
merged_df=pd.concat(gaps, ignore_index=True)
summed_df=merged_df.groupby('Gap Length (hours)')['Frequency'].sum().reset_index()
summed_df['rel_freq']=summed_df['Frequency']/summed_df['Frequency'].sum()
summed_df['Cumsum']=summed_df['rel_freq'].cumsum()*100


from matplotlib.cm import get_cmap
cmap = get_cmap('Blues')
plt.figure(figsize=cm2inch((9, 6.5)))
for i, station in enumerate(station_id_to_name.values()):

    df=gaps[i]
    color = cmap(i / len(station_id_to_name))
    plt.plot(df['Gap Length (hours)'], df['Cumsum'], lw=1,color=color)
    plt.xscale('log')

#dummie for creating label
plt.plot([],[], color='royalblue',lw=1, label='Individual station')
#plot cumulated frequency of ALL stations
plt.plot(summed_df['Gap Length (hours)'], summed_df['Cumsum'], lw=1,color='red', label='All stations')

plt.xlabel('Gap length [h]', fontsize='large')
plt.ylabel('Relative cumulative\nfrequency [%]', fontsize='large')
plt.tick_params(labelsize='medium')
plt.xlim(0.9,1.2*10**3)
plt.legend(fontsize='small')
plt.tight_layout()
plt.savefig(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\WL_Missing_plots\cum_freq_all_lines.png', dpi=600)

WL2=WL.copy()
WL2.columns=station_id_to_name.values()
WL2=WL2.rename(columns={'NS Tangeværket - v/Energimuseet': 'NS Tangeværket'})

fig, ax = plt.subplots(1, 1, figsize=cm2inch((10,15)))
msno.matrix(WL2, ax=ax, sparkline=False, fontsize=10)
# ax.set_yticklabels([])
y_ticks = [0, len(WL2)]
y_tick_labels = ['1985', '2023']
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels)

ax.tick_params('x', rotation=90)
ax.tick_params('y', labelsize=10)
plt.tight_layout()
# plt.savefig(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\WL_Missing_plots\missingno_matrix.png', dpi=600)

# msno.matrix(WL2[WL2.index>='2012'], ax=ax, sparkline=False, fontsize=12)
