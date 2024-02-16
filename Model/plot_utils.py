# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:34:24 2023

@author: Henriette
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta


# Thesis max size is 15.0 x 24.14 cm. Borrowed from https://zenodo.org/records/6726556 plot_utils
def cm2inch(*tupl: tuple):
    """
    Converts from cm to inches when defining figure sizes. Normal A4 page is 21 x 29.7 cm.
    
    Args
     tupl: Tuple containing (width, height) in cm
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def plot_losses(losslogpath, respath):
    # losses=pd.read_csv(r'C:\Users\henri\Documents\Universität\Masterthesis\Masterthesis\results\losslog.csv', sep=';', header=None)
    losses=pd.read_csv(losslogpath, sep=';', header=None)

    epoch=np.arange(len(losses)+1)
    epoch=epoch[1:]

    plt.figure()
    plt.plot(epoch,losses[0], label='train')
    plt.plot(epoch,losses[1], label='validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.savefig(respath)
    plt.close()


def plot_metrics_heatmap(rmse_df, PI_df, savepath):
    #Create heatmap of metrics results
    fig, axes = plt.subplots(1, 2, figsize=cm2inch((15, 9)),  sharey=True) 
    sns.heatmap(rmse_df, ax=axes[0], square=True, cmap='YlGn_r',cbar=False,fmt='.2f', annot=True, linewidth=.5)
    sns.heatmap(PI_df, ax=axes[1], square=True, cmap='YlGn_r', cbar=False, fmt='.0f',annot=True, linewidth=.5)
    # for i, _ in enumerate(axes): axes[i].xaxis.tick_top()
    axes[0].set_title('RMSE'); axes[1].set_title('PI')
    axes[0].set_ylabel('Training horizon [h]',fontsize='medium')
    axes[0].set_xlabel('Test horizon  [h]', fontsize='medium'); axes[1].set_xlabel('Test horizon  [h]',fontsize='medium')
    plt.savefig(savepath, dpi=600)
    plt.close()

