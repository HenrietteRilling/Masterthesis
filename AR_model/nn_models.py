# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:59:26 2023

@author: Henriette
"""
import torch

class get_FFNN(torch.nn.Module):
    def __init__(self,noinputs,hidden_size, output_size, bias):
        super(get_FFNN, self).__init__()
        #define the model layers to include
        self.linear1 = torch.nn.Linear(noinputs,hidden_size,bias=bias)
        torch.nn.init.normal_(self.linear1.weight,mean=0,std=0.001)
        self.linear2 = torch.nn.Linear(hidden_size,output_size,bias=bias)
        torch.nn.init.normal_(self.linear2.weight,mean=0,std=0.001)
        self.act = torch.nn.ReLU()
        self.dropout=torch.nn.Dropout(0.2)
        
        
    def forward(self, x):
        #say how the model layers will be applied
        # import pdb
        # pdb.set_trace()
        x = self.linear1(x)
        x = self.act(x)
        x= self.dropout(x)
        x = self.linear2(x)
        return x


###################Not yet implemented

class get_LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(get_LSTM, self).__init__()
        self.num_layers = num_layers #number of stacked LSTM layers
        self.input_size = input_size #number of expected features in the input x
        self.hidden_size = hidden_size #number of features in the hidden state h
        #define the model layers to include
        self.lstm1 = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=0.2) #Defintion of the LSTM
        self.fc = torch.nn.Linear(hidden_size, 1) #fully connected last layer, combines input to one output


    def forward(self, x):
        #define how the model layers will be applied
        #initialize hidden and cell states
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm1(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.fc(hn) #Final output
        return out.unsqueeze(-1) #unsqueeze adds another dimension of 1 to the tensor, necessary to have same shape as batched target data   

   
    
    
    
