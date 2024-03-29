# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:09:19 2023

@author: Henriette
"""

import os
import numpy as np
import pandas as pd
import pickle

import torch
#Set the number of CPU threads that PyTorch will use for parallel processing
torch.set_num_threads(8)

from utils import scale_data, timeseries_dataset_from_array, get_dataloader
from plot_utils import plot_losses, plot_metrics_heatmap
from nn_models import LSTM_AR
from trainer import Trainer
from metrics import rmse, PI
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import DatetimeTickFormatter



def run_LSTM(data, test_id, respath, train_period, val_period, test_period, training_horizons, imputation_horizons, epochs, b_size, nr_of_model_runs, neurons, num_lstm_layers, window_size):
    
    #################################################
    #split data in 70% train, 20% val, 10% test
    X_train=data[train_period[0]:train_period[1]]
    X_val=data[val_period[0]:val_period[1]]
    X_test=data[test_period[0]:test_period[1]]
    
    #scale and normalise such that all data has a range between [0,1], store scaler for rescaling
    X_train_sc, train_sc = scale_data(X_train)
    X_val_sc = train_sc.transform(X_val)
    X_test_sc = train_sc.transform(X_test)
    
    #get scaler only for waterlevel for unscaling predictions
    _, train_WL_sc=scale_data(X_train[[test_id]])
    
    
    #make resultpath
    if not os.path.exists(respath): os.makedirs(respath)
   
    #initialize dataframes for saving performance metrics
    rmse_df=pd.DataFrame(np.zeros((len(training_horizons), len(imputation_horizons))),index=training_horizons,columns=imputation_horizons)
    PI_df=pd.DataFrame(np.zeros((len(training_horizons), len(imputation_horizons))),index=training_horizons,columns=imputation_horizons)
    #initialize list to save predictions of best model for each th for plotting
    preds_test_all=[]
    for th in training_horizons:
        rmse_all_model_runs=np.zeros((nr_of_model_runs, len(imputation_horizons)))
        PI_all_model_runs=np.zeros((nr_of_model_runs, len(imputation_horizons)))
        #train several times for each forecast horizon
        best_val_loss=1.0
        for i in range(nr_of_model_runs):
            print(f'\nModel {i} of training horizon {th}') 
            
            #delete any existing losslog/files, to only save losses of current model run
            losslogpath=os.path.join(respath, f'losslog_{th}_{i}.csv')
            if os.path.exists(losslogpath): os.remove(losslogpath)
            #delete any existing weights, only save weights of current model run
            weightpath=os.path.join(respath, f'weights_{th}_{i}.pth')
            if os.path.exists(weightpath): os.remove(weightpath)
            
            ###################### Batch data ##########################################
            features_train, labels_train = timeseries_dataset_from_array(X_train_sc, window_size, th, AR=True)
            features_val, labels_val=timeseries_dataset_from_array(X_val_sc,  window_size, th, AR=True)

            #get data_loader for all data, data_loader is an torch iterable to be able to iterate over batches
            dataset_train, data_loader_train = get_dataloader(features_train, labels_train, batch_size=b_size)
            dataset_val, data_loader_val=get_dataloader(features_val, labels_val, batch_size=b_size, shuffle=False) #shuffle =False, as it is the set for validation???
            
            #############################################################
            #set up an autregressive model - the autoregressive model loop is defined in AR_model.py. The class in there imports a neural network configuration that is defined inside AR_nn_models.py, a feed forward model with 2 layers and as many neurons as defined in hidden size
            input_size=features_train.shape[-1]  #number of input features 
            # model=sampleLSTM_AR(input_size, neurons, num_lstm_layers, th)
            model=LSTM_AR(input_size, window_size, neurons, num_lstm_layers)
     
            trainer = Trainer(model,epochs, b_size, weightpath, losslogpath)
            cur_val_loss=trainer.fit(data_loader_train,data_loader_val)
            ###################
            
            #plot losses
            plot_losses(losslogpath, os.path.join(respath, f'losses_{th}_{i}.png'))
            
            # ###################################################################
            # ###################### Testing
            print(f'\nDone fitting model {i}')
            
            #load weights of best model
            model.load_state_dict(torch.load(weightpath))
            model.eval()
            
            for j, ih in enumerate(imputation_horizons):
                # print(f'Testing for imputation of {ih} hours')
                #create test features and label
                features_test, labels_test =timeseries_dataset_from_array(X_test_sc, window_size, ih, AR=True)
                dataset_test, data_loader_test=get_dataloader(features_test, labels_test, batch_size=b_size, shuffle=False)
                #generate predictions based on test set
                all_test_preds=[]
                for step, (x_batch_test, y_batch_test) in enumerate(data_loader_test):
                    #generate prediction
                    pred=model(x_batch_test)
                    #calculate metric for each batch, exlcude precipitation in target
                    all_test_preds.append(pred)
                
                ####################################
                #Metrics  
                #concat list elements along "time axis" 0
                preds_test=torch.cat(all_test_preds, 0).detach().numpy()
                #unscale all predicitons and labels
                preds_test_unsc=train_WL_sc.inverse_transform(preds_test[:,:,0])
                labels_test_unsc=train_WL_sc.inverse_transform(labels_test[:, :,0]) #include only water level
                features_test_unsc=train_WL_sc.inverse_transform(features_test[:,:,0]) #needed for PI, inlcude only water level
                #calculate metrics for current testhorizon
                rmse_all_model_runs[i, j]=rmse(labels_test_unsc[:len(preds_test),:], preds_test_unsc, savepath=None)
                PI_all_model_runs[i, j]=PI(labels_test_unsc[:len(preds_test),:], features_test_unsc[:len(preds_test),window_size-1:window_size], preds_test_unsc, savepath=None)

                if (cur_val_loss<best_val_loss) and (ih==imputation_horizons[-1]): 
                    best_val_loss=cur_val_loss
                    preds_test_to_keep=preds_test_unsc
                
                # Create a Bokeh figure for the last model
                if (i%nr_of_model_runs == 0) & (ih==imputation_horizons[-1]):
                    p = figure(x_axis_label='Date', y_axis_label='Water level [m]', title=f'Model {i} TH: {th} IH: {ih}')
                    
                    #plot prediction
                    p.line(pd.to_datetime(X_test.index)[window_size:-ih+1], preds_test_unsc[:,0], line_width=2, legend_label='Prediction', line_color='darkorange')            
                    # Plot observation
                    p.line(pd.to_datetime(X_test.index)[window_size:-ih+1], labels_test_unsc[:,0], line_width=2, legend_label='Observation', line_color='blue')
                    # Customize the plot
                    p.legend.location = 'top_left'
                    #control how dateticks are shown
                    p.xaxis.formatter = DatetimeTickFormatter(days="%m/%d",months="%m/%Y", years="%Y")
                    
                    # Save the plot
                    output_file(os.path.join(respath, f'preds_test_{th}_{ih}.html'))
                    save(p)
        #Calculate final metrics for each training horizon i.e., mean over all model runs
        rmse_df.loc[th][:]=rmse_all_model_runs.mean(axis=0)
        PI_df.loc[th][:]=PI_all_model_runs.mean(axis=0)
        preds_test_all.append(preds_test_to_keep)
        print(f'\nTrained and tested all models for training horizon: {th}')
            
        
    print('Yay! Finished simulation')

    #Crate a heatmap of the metrics
    plot_metrics_heatmap(rmse_df, PI_df, os.path.join(os.path.dirname(respath), f'{os.path.basename(respath)}_metrics.png'))
    #save metrics as csv file
    PI_df.columns = ['PI_' + str(col) for col in PI_df.columns]
    rmse_df.columns = ['RMSE_' + str(col) for col in rmse_df.columns]
    metrics_df=pd.concat((rmse_df, PI_df),axis=1)
    metrics_df.to_csv(os.path.join(os.path.dirname(respath),f'{os.path.basename(respath)}_metrics.csv'))
    
    #save predictions of current model configuration for plotting
    pickle_file_path = os.path.join(os.path.join(os.path.dirname(respath)),f'{os.path.basename(respath)}.pkl')
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump([preds_test_all, X_test[window_size:-imputation_horizons[-1]+1]], pickle_file)