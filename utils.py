
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from matplotlib.ticker import FormatStrFormatter
# import os
# import pandas as pd
# import xarray as xr
# import calendar
# from datetime import datetime
# # import sklearn
# # from sklearn import linear_model
# # from sklearn.metrics import mean_squared_error
# import scipy.stats
# from torch.utils.data import Dataset, DataLoader
# import torch.optim as optim
# import pickle
# from tqdm import tqdm
# # import globals_vars


# ######################## Define folders ########################
# CMIP6_ANOM_1M_FOLDER = 'datasets/CMIP6/CMIP6_regridded/anom_1m'
# CMIP6_ONI_FOLDER = 'datasets/CMIP6/CMIP6_regridded/oni'
# CMIP6_NINO34_FOLDER = 'datasets/CMIP6/CMIP6_regridded/nino34'

# HadISST_ANOM_1M_FOLDER = 'datasets/HadISST/HadISST_regridded/anom_1m'
# HadISST_ONI_FOLDER = 'datasets/HadISST/HadISST_regridded/oni'
# HadISST_NINO34_FOLDER = 'datasets/HadISST/HadISST_regridded/nino34'

# MODELS_ONI_FOLDER = 'saved_models/oni'
# MODELS_NINO34_FOLDER = 'saved_models/nino34'

# RESULTS_FOLDER = 'saved_results'
# RESULTS_ONI_FOLDER = 'saved_results/oni'
# RESULTS_NINO34_FOLDER = 'saved_results/nino34'

# RESULTS_NINO34_PF_FOLDER = os.path.join(RESULTS_NINO34_FOLDER, "persistent_forecast")
# RESULTS_NINO34_LR_FOLDER = os.path.join(RESULTS_NINO34_FOLDER, "linear_regression")
# RESULTS_NINO34_RR_FOLDER = os.path.join(RESULTS_NINO34_FOLDER, "ridge_regression")
# RESULTS_NINO34_CNN_FOLDER = os.path.join(RESULTS_NINO34_FOLDER, "cnn")
# os.makedirs(RESULTS_NINO34_CNN_FOLDER, exist_ok=True) 

# IMG_FOLDER = 'img/'
# SAVE_PLOTS_TO_DISK = True

# os.makedirs(MODELS_ONI_FOLDER, exist_ok=True) 
# os.makedirs(MODELS_NINO34_FOLDER, exist_ok=True) 

# convert longitude measurements from a 0° to 360° east system to standard longitude -180° to +180° system (spanning west to east)
def convert_lon(lon):
    if lon < 0 or lon > 360:
        raise Exception("Error: Longitude must be between 0 and 360")
    if lon > 180:
        return lon - 360
    else: 
        return lon

def lat_card(lat):
    if lat < -90 or lat > 90:
        raise Exception("Error: Latitude must be between -90 and +90")
    if lat < 0:
        card = 'S'
    else:
        card = 'N'
    return f"{abs(lat)}{card}"

def lon_card(lon):
    conv_lon = convert_lon(lon)
    if conv_lon < 0 or conv_lon == 360:
        card = 'W'
    else:
        card = 'E'
    return f"{abs(conv_lon)}{card}"


def add_cardinals_fname(min_lat, max_lat, min_lon, max_lon):
    """ Format coordinates for file names, e.g. Lat-5S-5N_Lon-170W-120W """
    if min_lon == 0 and max_lon == 360:
        return f"{lat_card(min_lat)}-{lat_card(max_lat)}-180W-180E"
    return f"{lat_card(min_lat)}-{lat_card(max_lat)}-{lon_card(min_lon)}-{lon_card(max_lon)}"

def add_cardinals_title(min_lat, max_lat, min_lon, max_lon):
    """ Format coordinates for plot titles, e.g. 5S-5N, 170W-120W """
    if min_lon == 0 and max_lon == 360:
        return f"{lat_card(min_lat)}-{lat_card(max_lat)}, 180W-180E"
    return f"{lat_card(min_lat)}-{lat_card(max_lat)}, {lon_card(min_lon)}-{lon_card(max_lon)}"






# def find_file(directory, source_id):
#     filenames = os.listdir(directory)

#     # Iterate over files int the directory and return the first match
#     for filename in filenames:
#         if source_id in filename:
#             filepath = os.path.join(directory, filename)
#             return filename, filepath
#     print(f"No file with source id {source_id} was found")
#     return None 


# def plot_nino_time_series(y, predictions, source_id, title):
#     """
#     inputs
#     ------
#         y           pd.Series : time series of the true Nino index
#         predictions np.array  : time series of the predicted Nino index (same
#                                 length and time as y)
#         titile                : the title of the plot

#     outputs
#     -------
#         None.  Displays the plot
#     """
#     predictions = pd.Series(predictions, index=y.index)
#     predictions = predictions.sort_index()
#     y = y.sort_index()
    
#     plt.figure(figsize=(6,1))  # Set the size of the figure
    
#     plt.plot(y, label=f'GT ({source_id})')
#     plt.plot(predictions, '--', label='ML Predictions')
#     # plt.legend(loc='best', fontsize=7)
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)

#     plt.title(title, fontsize=8)
#     plt.ylabel('ONI')
#     # plt.xlabel('Date')
#     plt.show()
#     plt.close()


# def assemble_predictors_predictands(source_id, target, start_date, end_date, lead_time, num_input_time_steps, lat_slice=None, lon_slice=None, verbose=False):
#     """
#     (inspired by CCAI AI for Climate science workshop - Forecasting the El Niño with Machine Learning)
    
#     inputs
#     ------
#       source_id         str : the source_id of the dataset to load
#       start_date        str : the start date from which to extract sst
#       end_date          str : the end date
#       lead_time         int : the number of months between each sst
#                               values and the target Oceanic Niño Index (ONI) 
#       num_input_time_steps int : the number of time steps to use for each
#                                  predictor sample
#       lat_slice           slice: the slice of latitudes to use
#       lon_slice           slice: the slice of longitudes to use
    
#     outputs
#     -------
#       Returns a tuple of the predictors (np array of sst temperature anomalies)
#       and the predictands (np array of the ONI index at the specified lead time).
    

#   """
#     data_format = "spatial"

#     start_date = pd.to_datetime(start_date)
#     end_date = pd.to_datetime(end_date)

    
#     #### SST VALUES ####
#     sst_filename, sst_filepath = find_file(CMIP6_ANOM_1M_FOLDER, source_id)
    
#     ds = xr.open_dataset(sst_filepath)
    

#     # Squeeze out dimensions of size one, if any (e.g. member_id, dcpp_init_year)
#     squeezed_ds = ds.squeeze()

#     # select sst anomalies for the required time period
#     sst_anom = squeezed_ds["tos_anom_1m"].sel(time=slice(start_date, end_date))

#     if lat_slice is not None:
#         sst_anom = sst_anom.sel(lat=lat_slice)
    
#     if lon_slice is not None:
#         sst_anom = sst_anom.sel(lon=lon_slice)

#     num_orig_samples = sst_anom.coords['time'].size

#     # converts sst_anom to (num_samples, num_input_time_steps, lat, lon) by stack arrays of (num_input_time_steps , lat, lon)
#     # e.g. np.stack([sst.values[0:2], sst.values[1:3], ...])
    
#     sst_anom = np.stack([sst_anom.values[n-num_input_time_steps : n] for n in range(num_input_time_steps, num_orig_samples+1)])
#     num_output_samples = sst_anom.shape[0]

#     if data_format == "flatten":
#         # reshape sst_anom into a 2D array: (num_output_samples, lat*lon*num_input_time_steps)
#         sst_anom = sst_anom.reshape(num_output_samples, -1)
#         # total_nb_entries = num_output_samples*sst_anom.shape[1]
    
#     # Returns the total number of elements in the input tensor.
#     total_nb_entries = sst_anom.size
    
#     # replace nan values with 0
#     nb_nan_entries = np.count_nonzero(np.isnan(sst_anom))
    
#     sst_anom[np.isnan(sst_anom)] = 0
    
#     if verbose: 
#         print(f"\n• Search for `{source_id}` SST file in {CMIP6_ANOM_1M_FOLDER}")
#         print(f"• Load {sst_filename}", end=" ")
#         print(f"({ds['time'][0].values.astype('datetime64[D]')} to {ds['time'][-1].values.astype('datetime64[D]')})")
#         print(f"• Slice the time period from {start_date.date()} to {end_date.date()}")
        
#         print(f"• Nb of original samples (nb months): {num_orig_samples}")
#         print(f"• Stack {num_input_time_steps} previous time steps for each sample")
#         print(f"• Nb of output samples, each containing {num_input_time_steps} time steps: {num_output_samples}")
#         if data_format == "flatten": 
#             print(f"• Reshape sst anomalies into a 2D {sst_anom.shape} array: {num_output_samples} months x {sst_anom.shape[1]:,} num_input_time_steps*lat*long values per month")
#         else:
#             print(f"• sst anomalies are a 4D {sst_anom.shape} array: {num_output_samples} months x {sst_anom.shape[1]:,} num_input_time_steps, {sst_anom.shape[2]:,} lat, {sst_anom.shape[3]:,} lon values per month")
#         print(f"• Replace {nb_nan_entries:,} nan values (continents) out of {total_nb_entries:,} with 0's ({nb_nan_entries/total_nb_entries:.2%})")
#         if data_format == "flatten":     
#             print(f"• The dimensions are: {sst_anom.shape}: {sst_anom.shape[0]} months, {sst_anom.shape[1]:,} num_input_time_steps*lat*long values per month")
#         else:
#             print(f"• The dimensions are: {sst_anom.shape}: {num_output_samples} months x {sst_anom.shape[1]:,} num_input_time_steps, {sst_anom.shape[2]:,} lat, {sst_anom.shape[3]:,} lon values per month")



#     ds.close()
#     squeezed_ds.close()


#     #### TARGET VALUES ####

#     if target == "oni":
#         target_folder = CMIP6_ONI_FOLDER
#         target_var_name = 'oni'
#     elif target == "nino34":
#         target_folder = CMIP6_NINO34_FOLDER
#         target_var_name = 'nino34'
#     else:
#         print(f"Unknown target {target}. Abort")
#         return

#     target_filename, target_filepath = find_file(target_folder, source_id)

#     df = pd.read_csv(target_filepath, sep='\t', index_col=0) 
#     df.index = pd.to_datetime(df.index)

#     target_ts = df[target_var_name]
        
#     lead_offset = pd.DateOffset(months=lead_time)
#     num_input_time_steps_offset = pd.DateOffset(months=num_input_time_steps-1)
#     start_date_plus_lead = start_date + lead_offset + num_input_time_steps_offset
#     end_date_plus_lead = end_date + lead_offset
    
#     target_ts = target_ts[start_date_plus_lead:end_date_plus_lead]

#     if verbose: 
#         print(f"\n• Search for `{source_id}` target file in {target_folder}")
#         print(f"• Load {target_filename}", end=" ")
#         print(f"({df.index[0].date()} to {df.index[-1].date()})")
#         print(f"• Slice the time period with {lead_time:02d} month(s) lead time and {num_input_time_steps_offset.months} months offset: from {start_date_plus_lead.date()} to {end_date_plus_lead.date()}")
#         print(f"• The dimensions are: {target_ts.shape}: {target_ts.shape[0]} months, 1 target value per month\n")

#     return sst_anom.astype(np.float32), target_ts.astype(np.float32)


# def train_network(model, criterion, optimizer, train_dataloader, val_dataloader,
#                   experiment_name, target, num_epochs=40, verbose=False):
#     """
#     inputs
#     ------
  
#         model             (nn.Module)   : the neural network architecture
#         criterion         (nn)          : the loss function (i.e. root mean squared error)
#         optimizer         (torch.optim) : the optimizer to use update the neural network
#                                           architecture to minimize the loss function
#         train_dataloader       (torch.utils.data.DataLoader): dataloader that loads the
#                                           predictors and predictands
#                                           for the train dataset
#         val_dataloader        (torch.utils.data. DataLoader): dataloader that loads the
#                                           predictors and predictands
#                                           for the validation dataset
#     outputs
#     -------
#         predictions (np.array), and saves the trained neural network as a .pt file
#     """

#     # select folder in which to save model depending on target
#     if target == "oni":
#         model_folder = MODELS_ONI_FOLDER
#     elif target == "nino34":
#         model_folder = MODELS_NINO34_FOLDER
#     else:
#         print(f"Unknown target {target}. Abort")
#         return
    
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     if verbose:
#         print("Prediction target: ", target)
#         print("Device used for training: ", device)
#     model = model.to(device)
#     best_loss = np.infty
#     train_losses, val_losses = [], []
#     best_epoch = 0

#     torch.save(model, f'{model_folder}/{experiment_name}.pt'.format(experiment_name))

#     for epoch in tqdm(range(num_epochs), leave=True, desc="Epochs", disable=False):
#         for mode, data_loader in [('train', train_dataloader), ('val', val_dataloader)]:
#             # Set the model to train mode to allow its weights to be updated while training
#             if mode == 'train':
#                 model.train()

#             # Set the model to eval model to prevent its weights from being updated while testing
#             elif mode == 'val':
#                 model.eval()

#             running_loss = 0.0
#             for i, data in enumerate(data_loader):
#                 # get a mini-batch of predictors and predictands
#                 batch_predictors, batch_predictands = data
#                 batch_predictands = batch_predictands.to(device)
#                 batch_predictors = batch_predictors.to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # calculate the predictions of the current neural network (FOREWARD PASS)

#                 # print("CNN's input tensor dimensions:", batch_predictors.shape)
#                 predictions = model(batch_predictors).squeeze()

#                 # quantify the quality of the predictions using a loss function (aka criterion) that is differentiable
#                 loss = criterion(predictions, batch_predictands)

#                 if mode == 'train':
#                     # the 'backward pass: calculates the gradients of each weight of the neural network with respect to the loss
#                     loss.backward()

#                     # the optimizer updates the weights of the neural network based on the gradients calculated above and the choice of optimization algorithm
#                     optimizer.step()

#                 # Add this batches loss to the running loss
#                 running_loss += loss.item()

#             if running_loss < best_loss and mode == 'val':
#                 # Save model (overwrite previous one if better running_loss)
#                 # print(f"     Val running_loss {running_loss:.2f} < best_loss {best_loss:.2f}, save {experiment_name}_checkpoint.pt model to disk")
#                 torch.save(model, f'{model_folder}/{experiment_name}.pt'.format(experiment_name))
#                 best_epoch = epoch + 1

#                 # update best_loss
#                 best_loss = running_loss

#             if verbose:
#                 print('{} Set: Epoch {:02d}. loss: {:3f}'.format(mode, epoch + 1, running_loss / len(data_loader)))
                
#             if mode == 'train':
#                 train_losses.append(running_loss / len(data_loader))
#             else:
#                 val_losses.append(running_loss / len(data_loader))
#     return train_losses, val_losses, best_epoch


# def infer(model, test_dataloader, verbose=True):
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     if verbose:
#         print("Device used for testing: ", device)
#         print("Size of dataloader: ", len(test_dataloader.dataset))
#         print("Number of batches in dataloader: ", len(test_dataloader))

#     model.eval()
#     model.to(device)

#     # calculates the predictions of the best saved model
#     predictions = np.asarray([])
#     for i, data in enumerate(test_dataloader):
#         batch_predictors, batch_predictands = data
#         batch_predictands = batch_predictands.to(device)
#         batch_predictors = batch_predictors.to(device)
#         batch_predictions = model(batch_predictors).squeeze()
#         if len(batch_predictions.size()) == 0:
#             batch_predictions = torch.Tensor([batch_predictions])

#         # Move the batch_predictions tensor to CPU, convert it to NumPy array, and concatenate it to the rest of the predictions
#         predictions = np.concatenate([predictions, batch_predictions.detach().cpu().numpy()])
#     return predictions


# def load_model(experiment_name, target):
#     if target == "oni":
#         print(f"• Loading ONI model {experiment_name}.pt")
#         model = torch.load(f'{MODELS_ONI_FOLDER}/{experiment_name}.pt')
#     elif target == "nino34":
#         print(f"• Loading Nino34 model {experiment_name}.pt")
#         model = torch.load(f'{MODELS_NINO34_FOLDER}/{experiment_name}.pt')
#     else:
#         print("• No matching target found. No model loaded")
#         return
#     return model


