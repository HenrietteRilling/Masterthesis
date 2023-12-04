# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:38:04 2023

@author: Henriette
"""
import os
import numpy as np


def rmse(y, y_hat, savepath):
    '''
    Function for calculating root mean squared error (RMSE)
    y= array, true values
    y_hat= array, predictions
    '''
    #calculate RMSE
    mse=np.mean((y-y_hat)**2)
    rmse=np.sqrt(mse)
    #save in logfile
    with open(savepath, 'a') as file:
        file.write(f'RMSE: {rmse}')
    return rmse


# =============================================================================
# Add Persistence index
# =============================================================================
    