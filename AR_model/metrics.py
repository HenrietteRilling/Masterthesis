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
    if savepath != None:
        #save in logfile
        with open(savepath, 'a') as file:
            file.write(f'{rmse};')
    else: return rmse


# =============================================================================
# Add Persistence index
# =============================================================================
    

def PI(y, ymin1, y_hat, savepath):
    '''
    Persistence Index
    compares the sum of squared error to the error that would occur if the value was forecast as the previous observed value.
    y = true observations
    ymin1 = true observations, shifted by one time step
    y_hat = predictions
    '''
    import pdb
    pdb.set_trace()
    SSE_pred=np.sum((y-y_hat)**2)
    SSE_prev=np.sum((y-ymin1)**2)
    PI=1-(SSE_pred/SSE_prev) 
    if savepath != None:
        #save in logfile
        with open(savepath, 'a') as file:
            file.write(f'{PI}\n')
    else: return PI