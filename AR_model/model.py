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
    self.model = get_FFNN(noinputs,hidden_size, output_size,True) #2=numberofinputs, hidden_size=number of neurons in hidden layer, bias=True
    self.output_size= output_size
  
  def forward(self, inputs):
      if self.output_size==1:
          
         #create first model prediction based on observation
         pred=self.model(inputs[:,0,:]) # in:[B,F], out:[B,1]
         #reshape prediction for matching dimensionality of labels
         result_tensor=torch.unsqueeze(pred, 1) #in: [B,1], out: [B, 1, 1]
        
         ###########################Autoregressive loop###################
         ####cycle through all time steps from 1 to windowsize, using the prediction from previous time step as model input
         t=1
         
         while t<inputs.size(1):
           #get precipitation at time step t
           prcp_for_this_step=torch.unsqueeze(inputs[:,t,1], 1) #[B,1]
           #merge precipitation and prediciton from time step t-1
           modelinput=torch.cat((result_tensor[:,-1], prcp_for_this_step),1)
           #generate new prediction for t+1
           pred=self.model(modelinput) #in: [B,F], out: [B,1]
           #save prediction
           result_tensor=torch.cat((result_tensor, torch.unsqueeze(pred, 1)),1)
           t+=1
         return result_tensor          
          
class sampleLSTM_AR(torch.nn.Module):
    def __init__(self, noinputfeatures, hidden_size, num_layers,training_horizon):
        super(sampleLSTM_AR, self).__init__()
        #get LSTM model defined in nn_model
        self.model=get_LSTM(noinputfeatures, hidden_size, num_layers)
        self.horizon=training_horizon
    
    def forward(self, features):
        #get windowsize for extracting features for first prediction
        #featureslength=(windowsize + forecast horizon-1), "future" values > windowsize are only needed for retrieving values of features in AR loop for timesteps t+1 ... t+h
        windowsize=features.shape[1]-self.horizon + 1
        #create first model prediciton based only on observations
        pred=self.model(features[:,:windowsize,:]) #in [B, W, F], out: [B, 1, 1]
        #reshape predictions
        result_tensor=pred
        
        t=1
        ##Autoregressive loop
        while t<self.horizon:
            #get w-t input from features
            features_for_this_step=features[:,t:windowsize+t,:]
            #replace water level observation with latest prediction 
            features_for_this_step[:,-t:,:1]=pred
            pred=self.model(features_for_this_step)
            result_tensor=torch.cat((result_tensor, pred),1)
            t+=1
        return result_tensor