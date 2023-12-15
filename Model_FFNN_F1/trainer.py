# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:46:23 2023

@author: Henriette
"""

import os,time
import torch

class Trainer():
    def __init__(self,model,epochs, batchsize, weightpath, losspath):
        self.model = model
        self.epochs = epochs
        self.batchsize=batchsize
        #
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        #
        self.loss_fn = torch.nn.MSELoss()
        #
        # if not os.path.exists(losspath): os.makedirs(losspath)
        self.losspath=losspath
        # if not os.path.exists(weightpath): os.makedirs(weightpath)
        self.weightpath=weightpath
        
    def _train_step(self, x, y):
        self.optimizer.zero_grad()
        #y doesn't need to be provided as input, first prediction is created based on true observation
        preds = self.model(x)
        # import pdb
        # pdb.set_trace()
        #y includes precipitation, has to be excluded for loss calculation
        #remove all features but water level (position 0) from y
        loss=self.loss_fn(preds, y[:,:,:1])
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def _test_step(self, x, y):
        with torch.no_grad():
            preds= self.model(x)
            #remove all features but water level (position 0) from y
            loss=self.loss_fn(preds, y[:,:,:1])
        return loss.item()
    

    def fit(self, data_loader_train, data_loader_val,startvalacc=1e10):
        train_loss_results = []; val_loss_results = []
        early_stopping_counter=0; early_stopping_criteria=20 #for early stopping, early_stopping_criteria= min number of epochs without change in val loss
        best_val=startvalacc
        training_start=time.time()
        for epoch in range(self.epochs):
            ######## Run training
            print("\nStart of epoch %d" % (epoch))
            start_time = time.time()
            # Iterate over the batches of the dataset.
            # import pdb
            # pdb.set_trace()
            self.model.train()
            train_losses = []
            for step, (x_batch_train, y_batch_train) in enumerate(data_loader_train):
               
                #skip training of batch, if batchsize is too small (applies for last batch at times, if lenght of dataset is not a multiplier of batch size)
                if x_batch_train.size(0)!=self.batchsize:
                    continue
                #train model
                loss_value = self._train_step(x_batch_train, y_batch_train)
                train_losses.append(loss_value)
                
            # Display metrics at the end of each epoch.
            train_acc = sum(train_losses)/len(train_losses)
            train_loss_results.append(train_acc)
            # print("Training acc over epoch: %.4f" % (float(train_acc),))
            # print("Training time: %.2fs" % (time.time() - start_time))
            
            ########
            # Run a validation loop at the end of each epoch.
            self.model.eval()
            val_losses = []
            for step, (x_batch_val, y_batch_val) in enumerate(data_loader_val):
                #skip training of batch, if batchsize is too small (applies for last batch at times, if lenght of dataset is not a multiplier of batch size)
                if x_batch_val.size(0)!=self.batchsize:
                    continue 
                loss_value = self._test_step(x_batch_val, y_batch_val)
                val_losses.append(loss_value)
            
            
            # print("Validation time: %.2fs" % (time.time() - start_time))
            
            val_acc = sum(val_losses)/len(val_losses)
            #safe model states, if validation loss improved
            if val_acc < best_val: 
                best_val = val_acc
                torch.save(self.model.state_dict(), self.weightpath)
                early_stopping_counter=0 #reset early stopping counter, as model improved
                
            #increase epoch counter if val loss didn't improve
            else:
                early_stopping_counter+=1    
            
            val_loss_results.append(val_acc)
            with open(self.losspath,'a') as f: f.write(str(train_acc)+';'+str(val_acc)+'\n')
            # print("Validation acc: %.4f" % (float(val_acc),))
            # print("Time taken: %.2fs" % (time.time() - start_time))
            
            #apply early stopping criteria
            if early_stopping_counter >=early_stopping_criteria:
                print(f'\nEarly stopping in {epoch}, no improvement of validation loss for {early_stopping_criteria} consecutive epochs.')
                print(f'Time taken: {time.time()-training_start}')
                print(f'Validation acc: {float(val_acc):.4f}')
                break
        print(f'\nFinished Training after {epoch} epochs.')
        print(f'Time taken: {time.time()-training_start}')
        print(f'Validation acc: {float(val_acc):.4f}')
