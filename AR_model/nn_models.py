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


class LSTM_cell_AR(torch.nn.Module):
    def __init__(self, input_size, window_size, hidden_size, num_layers) -> None:
        super().__init__()
        self.input_size=input_size
        self.window_size = window_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstmcell=torch.nn.LSTMCell(input_size, hidden_size)
        self.linear=torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out=[]
        x_pre = x[:, :self.window_size]
        x_post= x[:,self.window_size:]
        
        #initialize hidden and cell states
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32) #internal state
        
        #warm up LSTM
        _, (hn, cn)=self.lstm(x_pre, (h_0, c_0))
        out.append(self.linear(hn[-1]))
        
        # get last hidden state and cell state from lstm
        hn, cn = hn[-1], cn[-1]

        i=0
        ##Autoregressive loop, run one step predictions
        while i<(x_post.shape[1]):
            #get w-t input from features
            # feature_for_this_step=x_post[:,i,:]
            #replace water level observation with latest prediction 
            # feature_for_this_step[:,:1]=out[-1]
            feature_for_this_step=torch.cat((out[-1], x_post[:,i,1:]),1)
            hn, cn=self.lstmcell(feature_for_this_step, (hn, cn))
            out.append(self.linear(hn))
            i+=1
        return torch.stack(out, dim=1)
    
class LSTM_AR(torch.nn.Module):
    def __init__(self, input_size, window_size, hidden_size, num_layers) -> None:
        super().__init__()
        self.input_size=input_size
        self.window_size = window_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstmcell=torch.nn.LSTMCell(input_size, hidden_size)
        self.linear=torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        import pdb
        # pdb.set_trace()
        out=[]
        x_pre = x[:, :self.window_size]
        x_post= x[:,self.window_size:]
        
        #initialize hidden and cell states
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float32) #internal state
        
        #warm up LSTM
        _, (hn, cn)=self.lstm(x_pre, (h_0, c_0)) #input: x: [B,W,F], h_0,c_0: [num_layers, B, neurons], output: h_n, c_n: [1, B, neurons]
        out.append(self.linear(hn[-1])) #output: [B, 1]
        
        # get last hidden state and cell state from lstm
        # hn, cn = hn[-1], cn[-1] 

        i=0
        
        # pdb.set_trace()
        ##Autoregressive loop, run one step predictions
        while i<(x_post.shape[1]):
            #get w-t input from features
            # feature_for_this_step=x_post[:,i,:]
            #replace water level observation with latest prediction 
            # feature_for_this_step[:,:1]=out[-1]
            feature_for_this_step=torch.cat((out[-1], x_post[:,i,1:]),1).unsqueeze(1)
            _, (hn, cn)=self.lstm(feature_for_this_step, (hn, cn))
            out.append(self.linear(hn[-1]))
            i+=1
        return torch.stack(out, dim=1)


        
        
