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


  def forward(self, inputs,labels):
    batchsize = inputs.shape[0]
    ############
    #initialize the first prediction from the labels
    preds=labels[:,0,:]
    result_tensor=torch.unsqueeze(preds,1)
    ############################
    #cycle through all time steps from 1 to windowsize, using the prediction from previous time step as input
    n=1
    while n<inputs.shape[1]:  
        import pdb
        pdb.set_trace()
        input_for_this_step = inputs[:,n,:]
        modelinput=torch.cat([input_for_this_step,preds],1)
        preds=self.model(modelinput)
        result_tensor=torch.cat([result_tensor,torch.unsqueeze(preds,1)],1)
        #
        n+=1
    #
    return(result_tensor)