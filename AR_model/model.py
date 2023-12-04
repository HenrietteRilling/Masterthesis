# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:27:15 2023

@author: Henriette
"""

import numpy as np
import torch
from nn_models import get_FFNN, get_LSTM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class sampleFFNN_AR(torch.nn.Module):
  def __init__(self, noinputs, hidden_size, output_size):
    super(sampleFFNN_AR, self).__init__()
    #init - define things that will be needed for running the forward call below
    self.model = get_FFNN(noinputs,hidden_size, output_size,True) #2=numberofinputs, G=number of neurons in hidden layer, bias=True
    self.output_size= output_size
  
  def forward(self, inputs, labels):
      # import pdb
      # pdb.set_trace()
      if self.output_size==1:
          ###########################Autoregressive loop###################
          ####cycle through all time steps from 1 to windowsize, using the prediction from previous time step as additional model input feature
          #initialize first prediction from labels
          init_pred=labels[:,:1,0] #torch.Size([B, 1])
          
          # initialize result_tensor_AR with zeros
          result_tensor_AR = torch.zeros((init_pred.size(0), 1, 1)) #torch.Size([B, 1, 1])
    
          t=1
          while t <inputs.size(1):
              # import pdb
              # pdb.set_trace()
              #get features for current timestep t
              input_for_this_AR_step=inputs[:,t,:] #torch.Size([B, 2])
              #add prediction from timestep t-1 as input feature
              modelinput=torch.cat((input_for_this_AR_step, result_tensor_AR[:,-1,:]), 1) #torch.Size([B, 3])
              #generate new prediction for t+1
              preds=self.model(modelinput) #torch.Size([B, 1])
              #save prediction
              result_tensor_AR = torch.cat((result_tensor_AR, torch.unsqueeze(preds[:,:1], 1)),1)
              t+=1
          return result_tensor_AR
      
      elif self.output_size>1:
          #initialize first predictions from labels
          init_pred=torch.unsqueeze(labels[:,:self.output_size,0],1) #final shape: [B,1,H]
          result_tensor_AR=torch.zeros((init_pred.size()))

          t=1
          while t <inputs.size(1):
              #import pdb
              #pdb.set_trace()
              #get features for current timestep t
              input_for_this_AR_step=inputs[:,t,:] #torch.Size([B, F])
              #add prediction from timestep t-1 as input feature
              modelinput=torch.cat((input_for_this_AR_step, result_tensor_AR[:,-1,:]), 1) #torch.Size([B, F+H])
              preds=self.model(modelinput) #[B, H]
              #save prediction
              result_tensor_AR = torch.cat((result_tensor_AR, torch.unsqueeze(preds, 1)),1)
              t+=1
          return result_tensor_AR
          
          


#AR: receives prediction of time step before as input
class sampleFFNN_AR_2(torch.nn.Module):
  def __init__(self, noinputs, hidden_size, output_size):
    super(sampleFFNN_AR_2, self).__init__()
    #init - define things that will be needed for running the forward call below
    self.model = get_FFNN(noinputs,hidden_size, output_size,True) #2=numberofinputs, G=number of neurons in hidden layer, bias=True
    
  
  def forward(self, inputs, AR_input):
      ##################Add model output from timestep t-1 as additional feature to input tensor ###################
      # AR_preds have shape [B, H, 1], inputs [B, W, F], therefore reshaping of AR_preds necessary
      AR_feature=torch.zeros(inputs.size(0),inputs.size(1),1) #torch.Size([B, W, 1])
      AR_feature[:,-AR_input.size(1):,:]= AR_input

      #check for data gaps in the input
      # nan_mask=torch.isnan(inputs)
      # if torch.any(nan_mask):
      #     #Replace missing values with predictions from timestep before
      #     inputs[nan_mask]=AR_feature.expand(-1,-1,2)[nan_mask]

      
      #add AR_feature to input tensor
      modelinput=torch.cat((inputs, AR_feature),2) #torch.Size([B,W, F+1])

     
      #generate  new predictions
      preds=self.model(modelinput) #torch.Size([B, W, H])
      
      #reshape predictions to [B,H,1], only predictions from timestep t are taken as it is closest to the forecasted horizon t+1 to h
      preds=torch.unsqueeze(preds[:,-1,:],2)
      return preds      
          