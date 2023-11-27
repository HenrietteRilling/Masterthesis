# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:16:55 2023

@author: Henriette
"""

import os
import pandas as pd
import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from Data_loader import get_WL_data

if __name__ == "__main__":
    datafolder=r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\Data_WL'
    WL, labels, station_name_to_id, station_id_to_name = get_WL_data(datafolder)
    
    #calculate z-score
    zscore=(WL-WL.mean())/WL.std()
    #threshhold for detecting outliers
    threshold=3
 

# # =============================================================================
# #     Plot one plot for each station
# # =============================================================================
    
#     for col in WL.columns:
#         outliers=WL[col][np.abs(zscore[col]) > threshold]
#         mean=WL[col].mean()
#         #median=WL[col].median()
#         std=WL[col].std()
#         plt.figure()
#         ax = sns.kdeplot(data=WL[[col]],  fill=True, legend=False)
#         # Highlight outliers in red
#         ax.scatter(outliers, np.zeros_like(outliers), color='red', label='Extreme Values', zorder=5)
#         #Plot in addition some statistics
#         plt.axvline(mean, color='green', linestyle='--', label=f'Mean')
#         #plt.axvline(median, color='green', linestyle=':', label=f'Median')    
#         plt.axvline((mean+3*std), color='lightgreen', linestyle='--', label=f'3*Standard deviation')
#         plt.axvline((mean-3*std), color='lightgreen', linestyle='--')    
#         ax.set_xlabel('Water level [m]')
#         ax.set_title(station_id_to_name.get(col))
#         plt.legend()
#         # plt.savefig(os.path.join(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\WL_Distribution_plots'+f'\{col}.png'), dpi=300)
#         # plt.close()
        

        
# # =============================================================================
# # Plot all station in one plot with subplots
# # =============================================================================
#     rows=7
#     cols=3
#     fig, axes=plt.subplots(rows, cols, figsize=(16,10)) 
    
#     for i, col in enumerate(WL.columns):
#         #get row and column for current subplot
#         row=i//cols
#         column=i%cols
#         ax=axes[row, column]
        
#         #determine outliers and calculate statistics
#         outliers=WL[col][np.abs(zscore[col]) > threshold]
#         mean=WL[col].mean()
#         #median=WL[col].median()
#         std=WL[col].std()
#         sns.kdeplot(data=WL[[col]],  fill=True, legend=False, ax=ax)
#         # Highlight outliers in red
#         ax.scatter(outliers, np.zeros_like(outliers), color='red', label='Extreme Values', zorder=5)
#         #Plot in addition some statistics
#         ax.axvline(mean, color='green', linestyle='--', label=f'Mean')
#         #plt.axvline(median, color='green', linestyle=':', label=f'Median')    
#         ax.axvline((mean+3*std), color='lightgreen', linestyle='--', label=f'3*Standard deviation')
#         ax.axvline((mean-3*std), color='lightgreen', linestyle='--')    
#         #ax.set_xlabel('Water level [m]')
#         ax.set_ylabel('')
#         ax.set_title(station_id_to_name.get(col))


#     # Set shared x and y labels
#     fig.text(0.5, 0, 'Water level [m]', ha='center', fontsize=14)
#     fig.text(0, 0.5, 'Density', va='center', rotation='vertical', fontsize=14)
    
#     axes[0,2].legend()
#     plt.tight_layout()
#     # plt.savefig(os.path.join(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\WL_Distribution_plots\subplots.png'), dpi=300)
#     # plt.close() 
#     plt.show()   
    
    
    
# =============================================================================
#     Apply log transformation to the data and compare outcome
# =============================================================================
    
    WL_log=WL
    WL_log[WL_log<0]=np.nan
    WL_log = WL.apply(lambda x: np.log(x+1))
    zscore2=(WL_log-WL_log.mean())/WL_log.std()
    #threshhold for detecting outliers
    threshold=3
    
  # =============================================================================
  # Plot all station in one plot with subplots
  # =============================================================================
    rows=7
    cols=3
    fig, axes=plt.subplots(rows, cols, figsize=(16,10)) 
   
    for i, col in enumerate(WL.columns):
        #get row and column for current subplot
        row=i//cols
        column=i%cols
        ax=axes[row, column]
       
        #determine outliers and calculate statistics
        outliers=WL_log[col][np.abs(zscore2[col]) > threshold]
        mean=WL_log[col].mean()
        #median=WL[col].median()
        std=WL_log[col].std()
        sns.kdeplot(data=WL_log[[col]],  fill=True, legend=False, ax=ax)
        # Highlight outliers in red
        ax.scatter(outliers, np.zeros_like(outliers), color='red', label='Extreme Values', zorder=5)
        #Plot in addition some statistics
        ax.axvline(mean, color='green', linestyle='--', label=f'Mean')
        #plt.axvline(median, color='green', linestyle=':', label=f'Median')    
        ax.axvline((mean+3*std), color='lightgreen', linestyle='--', label=f'3*Standard deviation')
        ax.axvline((mean-3*std), color='lightgreen', linestyle='--')    
        #ax.set_xlabel('Water level [m]')
        ax.set_ylabel('')
        ax.set_title(station_id_to_name.get(col))


    # Set shared x and y labels
    fig.text(0.5, 0, 'Water level [m]', ha='center', fontsize=14)
    fig.text(0, 0.5, 'Density', va='center', rotation='vertical', fontsize=14)
   
    axes[0,2].legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(r'C:\Users\henri\Documents\Universität\Masterthesis\DMI_data\WL_Distribution_plots\subplots.png'), dpi=300)
    # plt.close() 
    plt.show()   
      
  