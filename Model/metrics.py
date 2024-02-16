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
    ymin1 = previous observed value
    y_hat = predictions
    '''
    #crate array with lenght of predictions, only containing first observation i.e., previous value is used as forecast for whole imputation horizon
    ymin1_0=np.tile(ymin1[:,:1],y.shape[1]) #np.tile(A, reps) A= value to be repeated, reps = number of repetitions
    SSE_pred=np.sum((y-y_hat)**2)
    SSE_prev=np.sum((y-ymin1_0)**2)
    PI=1-(SSE_pred/SSE_prev) 
    if savepath != None:
        #save in logfile
        with open(savepath, 'a') as file:
            file.write(f'{PI}\n')
    else: return PI
    
    