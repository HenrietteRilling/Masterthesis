# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:27:15 2023

@author: Henriette
"""

import numpy as np
import torch
from AR_nn_models import make_model_simple
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class samplemodel(torch.nn.Module):
  def __init__(self, noinputs, G):
    super(samplemodel, self).__init__()
    #init - define things that will be needed for running the forward call below
    self.model = make_model_simple(noinputs,G,True) #2=numberofinputs, G=number of neurons in hidden layer, bias=True
    self.ARpar = torch.nn.Parameter(torch.tensor(0,dtype=torch.float32))
    
  
  def forward(self, inputs, labels):
      # import pdb
      # pdb.set_trace()
      #generate first prediciton from the input
      preds=self.model(inputs)
      #initialize first prediction, we only take the last value as it is the next timestep in the future
      result_tensor=torch.unsqueeze(preds[:,-1,:],1)

      ############################
      #cycle through all time steps up until the maximal gap length/prediction horizon feeding predictions back
      n=0
      while n<labels.shape[1]-1: #-1 as for last prediction there is no more precipitation value in the features 
          input_for_this_step = inputs[:,1:,:]
          #add prcp observation of the respective timestep as input, prcp is "stored in the labels, we need to add a dimesion after extracting values
          preds_for_this_step=torch.cat([preds[:,-1:,:], torch.unsqueeze(labels[:,n,1:],1)],2)
          #check for missing values  in the input
          nan_mask=torch.isnan(input_for_this_step)
          if torch.any(nan_mask):
              #Replace missing values with predictions from timestep before
              input_for_this_step[nan_mask]=preds_for_this_step[nan_mask]  
          #concatenate observation and predictions
          modelinput=torch.cat([input_for_this_step,preds_for_this_step],1)
          #generate new prediction
          preds=self.model(modelinput)
          result_tensor=torch.cat([result_tensor,torch.unsqueeze(preds[:,-1,:],1)],1)
          n+=1
      return(result_tensor)     
  
  # #OLD forward function 
  # def forward(self, inputs,labels):
  #    ############
  #    #initialize the first prediction from the labels
  #    preds=labels[:,1,:]
  #    result_tensor=torch.unsqueeze(preds,1)
  #        ############################
  #    #cycle through all time steps from 1 to windowsize, using the prediction from previous time step as input
  #    n=0
  #    # import pdb
  #    # pdb.set_trace()
  #    while n<inputs.shape[1]-1: #-1 as for last prediciton there is no target to compare to 
  #        input_for_this_step = inputs[:,n,:]          
  #        #add prcp observation of the respective timestep as input
  #        preds_for_this_step=torch.cat([preds, inputs[:,n+1,1:]],1)
        
  #        #check for missing values  in the input
  #        nan_mask=torch.isnan(input_for_this_step)
  #        if torch.any(nan_mask):
  #            #Replace missing values with predictions from timestep before
  #            input_for_this_step[nan_mask]=preds_for_this_step[nan_mask]        
  #        modelinput=torch.cat([input_for_this_step,preds_for_this_step],1)
  #        preds=self.model(modelinput)
  #        result_tensor=torch.cat([result_tensor,torch.unsqueeze(preds,1)],1)
  #        #
  #        n+=1
  #    return(result_tensor)