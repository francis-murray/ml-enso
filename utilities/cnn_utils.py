import calendar
import os
import pickle
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.crs as ccrs
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import numpy.ma as ma
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xarray as xr
import xarray as xr
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MultipleLocator
from scipy.sparse import csr_matrix
from sklearn.covariance import empirical_covariance
from sklearn.preprocessing import StandardScaler
from sklearn import covariance
from sklearn.covariance import LedoitWolf
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

from typing import Dict

import scipy.stats

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import utils



MODELS_EC_FOLDER = 'saved_models/ec'
MODELS_NINO34_FOLDER = 'saved_models/nino34'
MODELS_ONI_FOLDER = 'saved_models/oni'

############### CNN ################

def train_cnn_epoch(model, device, train_dataloader, optimizer, epoch, criterion):
    model.train()        
    train_epoch_running_loss = 0.0
    train_epoch_loss_history = []
    train_total_samples = 0
    
    for i, batch in enumerate(train_dataloader):
        batch_predictors, batch_predictands = batch
        batch_predictands = batch_predictands.to(device)
        batch_predictors = batch_predictors.to(device)

        batch_size = batch_predictors.size(0)
        train_total_samples += batch_size
        
        optimizer.zero_grad() # zero the parameter gradients

        # Forward pass
        batch_predictions = model(batch_predictors).squeeze()
        batch_loss = criterion(batch_predictions, batch_predictands)
        # print(f"Train batch loss ({batch_size} samples): {batch_loss}")

        # Backward pass and optimization
        batch_loss.backward() 
        optimizer.step()

        # Add this batch loss to the running loss and append to history
        train_epoch_running_loss += batch_loss.item() * batch_size  # Weight by batch size
        train_epoch_loss_history.append(batch_loss.item())
        
    # train_epoch_avg_loss = train_epoch_running_loss / len(train_dataloader) # average loss per batch
    train_epoch_avg_loss = train_epoch_running_loss  / train_total_samples  # Weighted average loss per sample

    # print(f"-----Train epoch_avg_loss: {train_epoch_avg_loss}")
    return train_epoch_avg_loss, train_epoch_loss_history


@torch.no_grad()
def validate_cnn_epoch(model, device, val_dataloader, criterion):
    model.eval()
    val_epoch_running_loss = 0.0
    val_epoch_loss_history = []
    val_total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            batch_predictors, batch_predictands = batch
            batch_predictands = batch_predictands.to(device)
            batch_predictors = batch_predictors.to(device)

            batch_size = batch_predictors.size(0)          
            val_total_samples += batch_size

            # Forward pass
            batch_predictions = model(batch_predictors).squeeze()
            batch_loss = criterion(batch_predictions, batch_predictands)
            # print(f"Val batch_loss ({batch_size} samples): {batch_loss}")

            
            # Add this batch loss to the running loss and append to history
            val_epoch_running_loss += batch_loss.item() * batch_size  # Weight by batch size
            val_epoch_loss_history.append(batch_loss.item())
            
    # val_epoch_avg_loss = val_epoch_running_loss / len(val_dataloader) # average loss per batch
    val_epoch_avg_loss = val_epoch_running_loss  / val_total_samples  # Weighted average loss per sample
    # print(f"-----Val epoch_avg_loss: {val_epoch_avg_loss}")
    return val_epoch_avg_loss, val_epoch_loss_history


def train_cnn_network(model, criterion, optimizer, train_dataloader, val_dataloader,
                  experiment_name, target, num_epochs=40, verbose=False, save_model=True):
    """
    inputs
    ------
        model             (nn.Module)   : the neural network architecture
        criterion         (nn)          : the loss function (i.e. root mean squared error)
        optimizer         (torch.optim) : the optimizer to use update the neural network
                                          architecture to minimize the loss function
        train_dataloader       (torch.utils.data.DataLoader): dataloader that loads the
                                          predictors and predictands
                                          for the train dataset
        val_dataloader        (torch.utils.data. DataLoader): dataloader that loads the
                                          predictors and predictands
                                          for the validation dataset
    outputs
    -------
        train_losses, train_loss_history, val_losses, best_epoch 
        and saves the trained neural network as a .pt file
    """

    # select folder in which to save model depending on target
    if target == "oni":
        model_folder = MODELS_ONI_FOLDER
    elif target == "nino34":
        model_folder = MODELS_NINO34_FOLDER
    elif target == "E":
        model_folder = MODELS_EC_FOLDER
    elif target == "C":
        model_folder = MODELS_EC_FOLDER
    else:
        print(f"Unknown target {target}. Abort")
        return
    
    if "CNN" in experiment_name:
        model_type = "cnn"
    elif "GCN" in experiment_name:
        model_type = "gcn"
    else:
        print(f"Unknown model type in experiment_name {experiment_name}. Abort")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if verbose:
        print("Prediction target: ", target)
        print("Device used for training: ", device)
    model = model.to(device)

    # create 1 folder for all models of the same architecture with different leads
    model_subfolder = experiment_name.split("_Lead")[0]
    os.makedirs(f'{model_folder}/{model_type}/{model_subfolder}', exist_ok=True) 
    save_path = f'{model_folder}/{model_type}/{model_subfolder}/{experiment_name}.pt'        
    print(f"Save model to save_path: {save_path}")
        
    # save initial model here in case it doesn't improve
    save_model_state(model, optimizer, save_path)

    # Print the model architecture to a text file
    with open(f'{model_folder}/{model_type}/{model_subfolder}/model_architecture.txt', 'w') as f:
        f.write(str(model))

    best_epoch = 0
    best_loss = np.infty
    train_losses, val_losses = [], []
    train_loss_history, val_loss_history = [], []
    
        
            
    for epoch in tqdm(range(num_epochs), leave=False, desc="Epochs", disable=False):
        # print(f"    • Epoch: {epoch}")

        ##### Taining phase #####
        train_epoch_avg_loss, train_epoch_loss_history = train_cnn_epoch(model, device, train_dataloader, optimizer, epoch, criterion)
        train_losses.append(train_epoch_avg_loss)
        train_loss_history.extend(train_epoch_loss_history)

        ##### Validation phase #####
        val_epoch_avg_loss, val_epoch_loss_history = validate_cnn_epoch(model, device, val_dataloader, criterion)
        val_losses.append(val_epoch_avg_loss)
        val_loss_history.extend(val_epoch_loss_history)
                
        if val_epoch_avg_loss < best_loss:
            best_loss = val_epoch_avg_loss
            best_epoch = epoch + 1
            if save_model:
                save_model_state(model, optimizer, save_path)
                
        if verbose:
            print(f'        • Epoch {epoch + 1}/{num_epochs}. Train Avg Loss: {train_epoch_avg_loss:.4f}, Val Avg Loss: {val_epoch_avg_loss:.4f}')

    return train_losses, train_loss_history, val_losses, best_epoch



def infer_cnn(model, test_dataloader, verbose=True):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if verbose:
        print("Device used for testing: ", device)
        print("Size of dataloader: ", len(test_dataloader.dataset))
        print("Number of batches in dataloader: ", len(test_dataloader))


    model.to(device)
    model.eval()

    # calculates the predictions of the best saved model
    predictions = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            batch_predictors, batch_predictands = batch
            batch_predictands = batch_predictands.to(device)
            batch_predictors = batch_predictors.to(device)
            batch_predictions = model(batch_predictors).squeeze()
          
            if len(batch_predictions.size()) == 0:
                batch_predictions = batch_predictions.unsqueeze(0)
            
            # Append batch predictions to the list
            predictions.append(batch_predictions.detach().cpu().numpy())
            
    # Concatenate all batch predictions into a single array
    predictions = np.concatenate(predictions)
    return predictions



def save_model_state(model, optimizer, save_path):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, save_path)
    # print(f"Saved model to {save_path}")

#####################################