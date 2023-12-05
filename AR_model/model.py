# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:27:15 2023

@author: Henriette
"""

import numpy as np
import torch
from nn_models import get_FFNN
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
          

          