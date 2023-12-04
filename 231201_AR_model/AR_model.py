# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:27:15 2023

@author: Henriette
"""

import numpy as np
import torch
from AR_nn_models import get_FFNN_AR, get_LSTM_AR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class sampleFFNN_AR(torch.nn.Module):
  def __init__(self, noinputs, hidden_size, output_size):
    super(sampleFFNN_AR, self).__init__()
    #init - define things that will be needed for running the forward call below
    self.model = get_FFNN_AR(noinputs,hidden_size, output_size,True) #2=numberofinputs, G=number of neurons in hidden layer, bias=True
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
              #get features for current timestep t
              input_for_this_AR_step=inputs[:,t,:] #torch.Size([B, F])
              #add prediction from timestep t-1 as input feature
              modelinput=torch.cat((input_for_this_AR_step, result_tensor_AR[:,-1,:]), 1) #torch.Size([B, F+H])
              preds=self.model(modelinput) #preds shape: [B, H]
              #save prediction
              result_tensor_AR = torch.cat((result_tensor_AR, torch.unsqueeze(preds, 1)),1)
              t+=1
          return result_tensor_AR
          
          


#AR: receives prediction of time step before as input
class sampleFFNN_AR_2(torch.nn.Module):
  def __init__(self, noinputs, hidden_size, output_size):
    super(sampleFFNN_AR_2, self).__init__()
    #init - define things that will be needed for running the forward call below
    self.model = get_FFNN_AR(noinputs,hidden_size, output_size,True) #2=numberofinputs, G=number of neurons in hidden layer, bias=True
    
  
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
          

#simple LSTM model             

class sampleLSTM_AR(torch.nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super(sampleLSTM_AR, self).__init__()
    #init - define things that will be needed for running the forward call below
    self.model = get_LSTM_AR(input_size, hidden_size, num_layers) #inputsize= number of features,hidden size=number of neurons/features in the hidden size, num_layers=number of stacked LSTM layers

    
  
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
  
  
      ##########################Future predictions######################### (in FFNN)
      
      # if labels.shape[1]>1:
      #     #initialize with last prediction from AR
      #     result_tensor_future=result_tensor_AR[:,-1:,:] #torch.Size([B,1,1])
      #     n=0
          
      #     #create as many predictions as defined in horizon (len of feature sequence), -1 as for last prediction there is no more precipitation value in the features
      #     while n<labels.shape[1]-1:
      #         #shift input by one step, dropping the oldest time record
      #         input_for_this_fut_step=inputs[:,(n+1):,:]
      #         #take prediction from latest timestep and add precipitation observation
      #         preds_for_this_fut_step=torch.cat([result_tensor_future[:,-(n+1):,:], labels[:,:n+1,1:]],2)
      #         #create input tensor with prediction as latest timestep
      #         modelinput_fut=torch.cat([input_for_this_fut_step,preds_for_this_fut_step],1)
      #         #Add predictions from AR loop as additional model input feature TODO: should AR also be updated?????
      #         modelinput_fut=torch.cat([modelinput_fut, result_tensor_AR],2)
      #         #generate prediction for next time step
      #         preds_fut=self.model(modelinput_fut)
      #         #save prediction ("earliest prediction" produced from last time step in input sequence)
      #         result_tensor_future=torch.cat([result_tensor_future,torch.unsqueeze(preds_fut[:,-1,:1],1)],1)
              
      #         n+=1
      #     # import pdb
      #     # pdb.set_trace()
      #     return result_tensor_future
      # else: return result_tensor_AR
