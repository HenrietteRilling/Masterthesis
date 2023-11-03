# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:46:23 2023

@author: Henriette
"""

import os,time
import torch

class Trainer():
    def __init__(self,model,epochs,outpath):
        self.model = model
        self.epochs = epochs
        #
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        #
        self.loss_fn = torch.nn.MSELoss()
        #
        if not os.path.exists(outpath): os.makedirs(outpath)
        self.outpath=outpath
        
    def _train_step(self, x, y):
        self.optimizer.zero_grad()
        #y needs to provided as input, but only the first value is used in each time step
        preds = self.model(x,y)
        loss=self.loss_fn(preds, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def _test_step(self, x, y):
        with torch.no_grad():
            preds= self.model(x,y)
            loss=self.loss_fn(preds, y)
        return loss.item()
    

    def fit(self, data_loader_train, data_loader_val,startvalacc=1e10):
        train_loss_results = []; val_loss_results = []
        best_val=startvalacc
        for epoch in range(self.epochs):
            ########
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            #initialize the data generators
            #To Do: possibly create Dataset class, that includes dataloader calling + windowing of data??
            # train_gen=multi_window_train.data_generator(runendless = False)
            # val_gen=multi_window_val.data_generator(runendless = False)
            # Iterate over the batches of the dataset.
            # import pdb
            # pdb.set_trace()
            self.model.train()
            train_losses = []; train_details = []
            for step, (x_batch_train, y_batch_train) in enumerate(data_loader_train):
                loss_value = self._train_step(x_batch_train, y_batch_train)
                train_losses.append(loss_value)
            # Display metrics at the end of each epoch.
            train_acc = sum(train_losses)/len(train_losses)
            train_loss_results.append(train_acc)
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            print("Training time: %.2fs" % (time.time() - start_time))
            ########
            # Run a validation loop at the end of each epoch.
            self.model.eval()
            val_losses = []
            for step, (x_batch_val, y_batch_val) in enumerate(data_loader_val):
                loss_value = self._test_step(x_batch_val, y_batch_val)
                val_losses.append(loss_value)
                #print(step)
            print("Validation time: %.2fs" % (time.time() - start_time))
            val_acc = sum(val_losses)/len(val_losses)
            if val_acc < best_val: 
                best_val = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.outpath,'weights.pth'))
            val_loss_results.append(val_acc)
            #
            with open(os.path.join(self.outpath,'losslog.csv'),'a') as f: f.write(str(train_acc)+';'+str(val_acc)+'\n')
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))
