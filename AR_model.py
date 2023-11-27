# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:27:15 2023

@author: Henriette
"""

import numpy as np
import torch
from AR_nn_models import make_model_simple, LSTM_simple
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class samplemodel(torch.nn.Module):
  def __init__(self, noinputs, G):
    super(samplemodel, self).__init__()
    #init - define things that will be needed for running the forward call below
    self.model = make_model_simple(noinputs,G,True) #2=numberofinputs, G=number of neurons in hidden layer, bias=True
    
  
  def forward(self, inputs, labels):
  
      ###########################Autoregressive loop###################
      ####cycle through all time steps from 1 to windowsize, using the prediction from previous time step as additional model input feature
      #initialize first prediction
      init_pred=inputs[:,:1,0] #torch.Size([B, 1])
      #add dimension to match shape of model predictions
      result_tensor_AR=torch.unsqueeze(init_pred, 2) #torch.Size([B, 1, 1])
      ############Note: it's maybe possible to drop unsqueezing, but then adjustment of loss calculation necessary
      t=1
      while t <inputs.size(1):
          #get features for current timestep t
          input_for_this_AR_step=inputs[:,t,:] #torch.Size([B, 2])
          #add prediction from timestep t-1 as input feature
          modelinput=torch.cat((input_for_this_AR_step, result_tensor_AR[:,-1,:]), 1) #torch.Size([B, 3])
          #generate new prediction for t+1
          preds=self.model(modelinput) #torch.Size([B, 1])
          #save prediction
          result_tensor_AR = torch.cat((result_tensor_AR, torch.unsqueeze(preds, 1)),1)
          t+=1
      # import pdb
      # pdb.set_trace()
      ##########################Future predictions#########################
      
      if labels.shape[1]>1:
          #initialize with last prediction from AR
          result_tensor_future=result_tensor_AR[:,-1:,:] #torch.Size([B,1,1])
          n=0
          
          #create as many predictions as defined in horizon (len of feature sequence), -1 as for last prediction there is no more precipitation value in the features
          while n<labels.shape[1]-1:
              #shift input by one step, dropping the oldest time record
              input_for_this_fut_step=inputs[:,(n+1):,:]
              #take prediction from latest timestep and add precipitation observation
              preds_for_this_fut_step=torch.cat([result_tensor_future[:,-(n+1):,:], labels[:,:n+1,1:]],2)
              #create input tensor with prediction as latest timestep
              modelinput_fut=torch.cat([input_for_this_fut_step,preds_for_this_fut_step],1)
              #Add predictions from AR loop as additional model input feature TODO: should AR also be updated?????
              modelinput_fut=torch.cat([modelinput_fut, result_tensor_AR],2)
              #generate prediction for next time step
              preds_fut=self.model(modelinput_fut)
              #save prediction
              result_tensor_future=torch.cat([result_tensor_future,torch.unsqueeze(preds_fut[:,-1,:],1)],1)
              
              n+=1
          # import pdb
          # pdb.set_trace()
          return result_tensor_future
 

#simple LSTM model             

class sampleLSTMmodel(torch.nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super(sampleLSTMmodel, self).__init__()
    #init - define things that will be needed for running the forward call below
    self.model = LSTM_simple(input_size, hidden_size, num_layers) #inputsize= number of features,hidden size=number of neurons/features in the hidden size, num_layers=number of stacked LSTM layers

    
  
  def forward(self, inputs, labels):
  
      ###########################Autoregressive loop###################
      ####cycle through all time steps from 1 to windowsize, using the prediction from previous time step as additional model input feature
      #initialize first prediction
      init_pred=inputs[:,:1,0] #torch.Size([B, 1])
      #add dimension to match shape of model predictions
      result_tensor_AR=torch.unsqueeze(init_pred, 2) #torch.Size([B, 1, 1])
      ############Note: it's maybe possible to drop unsqueezing, but then adjustment of loss calculation necessary
      t=1
      while t <inputs.size(1):
          #get features for current timestep t
          input_for_this_AR_step=inputs[:,t,:] #torch.Size([B, 2])
          #add prediction from timestep t-1 as input feature
          modelinput=torch.cat((input_for_this_AR_step, result_tensor_AR[:,-1,:]), 1) #torch.Size([B, F])
          #generate new prediction for t+1
          preds=self.model(torch.unsqueeze(modelinput, 1)) #torch.Size([B, S, F])
          #save prediction
          result_tensor_AR = torch.cat((result_tensor_AR, preds),1)
          t+=1
          
      ##########################Future predictions#########################
      
      if labels.shape[1]>1:
          #initialize with last prediction from AR
          result_tensor_future=result_tensor_AR[:,-1:,:] #torch.Size([B,1,1])
          n=0
          
          #create as many predictions as defined in horizon (len of feature sequence), -1 as for last prediction there is no more precipitation value in the features
          while n<labels.shape[1]-1:
              #shift input by one step, dropping the oldest time record
              input_for_this_fut_step=inputs[:,(n+1):,:]
              #take prediction from latest timestep and add precipitation observation
              preds_for_this_fut_step=torch.cat([result_tensor_future[:,-(n+1):,:], labels[:,:n+1,1:]],2)
              #create input tensor with prediction as latest timestep
              modelinput_fut=torch.cat([input_for_this_fut_step,preds_for_this_fut_step],1)
              #Add predictions from AR loop as additional model input feature TODO: should AR also be updated?????
              modelinput_fut=torch.cat([modelinput_fut, result_tensor_AR],2)
              #generate prediction for next time step
              preds_fut=self.model(modelinput_fut) #torch.Size([B,S,F])
              #save prediction
              result_tensor_future=torch.cat([result_tensor_future,torch.unsqueeze(preds_fut[:,-1,:],1)],1)
              
              n+=1
          return result_tensor_future



        
###old forward loop for model without AR but with future predictions

    # #generate first prediciton from the input
    #   preds=self.model(inputs)
    #   #initialize first prediction, we only take the last value as it is the next timestep in the future
    #   result_tensor=torch.unsqueeze(preds[:,-1,:],1)

    #   ############################
    #   #cycle through all time steps up until the maximal gap length/prediction horizon feeding predictions back
    #   n=0
    #   while n<labels.shape[1]-1: #-1 as for last prediction there is no more precipitation value in the features 
    #       input_for_this_step = inputs[:,1:,:]
    #       #add prcp observation of the respective timestep as input, prcp is "stored in the labels, we need to add a dimesion after extracting values
    #       preds_for_this_step=torch.cat([preds[:,-1:,:], torch.unsqueeze(labels[:,n,1:],1)],2)
    #       #check for missing values  in the input
    #       nan_mask=torch.isnan(input_for_this_step)
    #       if torch.any(nan_mask):
    #           #Replace missing values with predictions from timestep before
    #           input_for_this_step[nan_mask]=preds_for_this_step[nan_mask]  
    #       #concatenate observation and predictions
    #       modelinput=torch.cat([input_for_this_step,preds_for_this_step],1)
    #       #generate new prediction
    #       preds=self.model(modelinput)
    #       result_tensor=torch.cat([result_tensor,torch.unsqueeze(preds[:,-1,:],1)],1)
    #       n+=1
    #   return(result_tensor)     
  
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