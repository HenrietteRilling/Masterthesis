# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:59:26 2023

@author: Henriette
"""
import torch

class make_model_simple(torch.nn.Module):
    def __init__(self,noinputs,G,bias):
        super(make_model_simple, self).__init__()
        #define the model layers to include
        self.linear1 = torch.nn.Linear(noinputs,G,bias=bias)
        torch.nn.init.normal_(self.linear1.weight,mean=0,std=0.001)
        self.linear2 = torch.nn.Linear(G,1,bias=bias)
        torch.nn.init.normal_(self.linear2.weight,mean=0,std=0.001)
        self.act = torch.nn.ReLU()
        
        
    def forward(self, x):
        #say how the model layers will be appliied
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x
