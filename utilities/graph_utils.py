import calendar
import os
import pickle


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
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime
from dateutil.relativedelta import relativedelta
from datetime import timedelta

from typing import Dict

import scipy.stats

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import utils


MODELS_EC_FOLDER = 'saved_models/ec'
MODELS_NINO34_FOLDER = 'saved_models/nino34'
MODELS_ONI_FOLDER = 'saved_models/oni'

RESULTS_EC_FOLDER = 'saved_results/ec'
RESULTS_FOLDER = 'saved_results'
RESULTS_NINO34_FOLDER = 'saved_results/nino34'
RESULTS_ONI_FOLDER = 'saved_results/oni'







CMIP6_ANOM_1M_1X1D_FOLDER = 'datasets/CMIP6/CMIP6_regridded/anom_1m'
CMIP6_ANOM_1M_4X4D_FOLDER = 'datasets/CMIP6/CMIP6_regridded/anom_1m_4x4deg'
CMIP6_ANOM_1M_5X5D_FOLDER = 'datasets/CMIP6/CMIP6_regridded/anom_1m_5x5deg'
CMIP6_EC_FOLDER = 'datasets/CMIP6/CMIP6_regridded/ec_indices'
CMIP6_NINO34_FOLDER = 'datasets/CMIP6/CMIP6_regridded/nino34'
CMIP6_ONI_FOLDER = 'datasets/CMIP6/CMIP6_regridded/oni'

HadISST_ANOM_1M_1X1D_FOLDER = 'datasets/HadISST/HadISST_regridded/anom_1m'
HadISST_ANOM_1M_4X4D_FOLDER = 'datasets/HadISST/HadISST_regridded/anom_1m_4x4deg'
HADISST_ANOM_1M_5X5D_FOLDER = 'datasets/HadISST/HadISST_regridded/anom_1m_5x5deg'
HADISST_EC_FOLDER = 'datasets/HadISST/HadISST_regridded/ec_indices'
HADISST_NINO34_FOLDER = 'datasets/HadISST/HadISST_regridded/nino34'
HADISST_ONI_FOLDER = 'datasets/HadISST/HadISST_regridded/oni'


GODAS_ANOM_1M_1X1D_FOLDER = 'datasets/GODAS/GODAS_regridded/anom_1m'
GODAS_ANOM_1M_4X4D_FOLDER = 'datasets/GODAS/GODAS_regridded/anom_1m_4x4deg'
GODAS_ANOM_1M_5X5D_FOLDER = 'datasets/GODAS/GODAS_regridded/anom_1m_5x5deg'
GODAS_EC_FOLDER = 'datasets/GODAS/GODAS_regridded/ec_indices'
GODAS_NINO34_FOLDER = 'datasets/GODAS/GODAS_regridded/nino34'
GODAS_ONI_FOLDER = 'datasets/GODAS/GODAS_regridded/oni'

def add_node_features(G, data_array, datetime, feature_name="x"):
    count = 0
    reverse_pos_dict = get_reverse_pos_dict(G)
    data_array_sel = data_array.sel(time=datetime)

    # Add node values (features) from a DataArray
    for lat in data_array_sel.lat.values:
        for lon in data_array_sel.lon.values:
            node_id = reverse_pos_dict.get((lon, lat))
            if node_id is not None:
                G.nodes[node_id][feature_name] = [data_array_sel.sel(lat=lat, lon=lon).item()]
                count += 1
            # else:
            #     print(f"{(lon, lat)} is not in graph")
    print(f"Added features to {count} nodes")
    return G


def assemble_graph_predictors_predictands(source_id, resolution, target, start_date, end_date, lead_time, 
                                          num_input_time_steps, lat_slice=None, lon_slice=None, data_format="spatial", 
                                          use_pca=False, pca_components=32, fill_nan=False, remove_win=False, verbose=False):
    """
    (inspired by CCAI AI for Climate science workshop - Forecasting the El Niño with Machine Learning)
    
    inputs
    ------
      source_id         str : the source_id of the dataset to load
      start_date        str : the start date from which to extract sst
      end_date          str : the end date
      lead_time         int : the number of months between each sst
                              values and the target Oceanic Niño Index (ONI) 
      num_input_time_steps int : the number of time steps to use for each
                                 predictor sample
      lat_slice           slice: the slice of latitudes to use
      lon_slice           slice: the slice of longitudes to use
    
    outputs
    -------
      Returns a tuple of the predictors (np array of sst temperature anomalies)
      and the predictands (np array of the ONI index at the specified lead time).
    

  """
  
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    #### SST VALUES ####
    if resolution == 5:
        if source_id.split("-")[0] == "HADISST":
            input_folder = HADISST_ANOM_1M_5X5D_FOLDER
        elif source_id.split("-")[0] == "GODAS":
            input_folder = GODAS_ANOM_1M_5X5D_FOLDER
        else:
            input_folder = CMIP6_ANOM_1M_5X5D_FOLDER
    elif resolution == 4:
        if source_id.split("-")[0] == "HADISST":
            input_folder = HadISST_ANOM_1M_4X4D_FOLDER
        elif source_id.split("-")[0] == "GODAS":
            input_folder = GODAS_ANOM_1M_4X4D_FOLDER
        else: 
            input_folder = CMIP6_ANOM_1M_4X4D_FOLDER            
    elif resolution == 1:
        if source_id.split("-")[0] == "HADISST":
            input_folder = HadISST_ANOM_1M_1X1D_FOLDER
        elif source_id.split("-")[0] == "GODAS":
            input_folder = GODAS_ANOM_1M_1X1D_FOLDER
        else: 
            input_folder = CMIP6_ANOM_1M_1X1D_FOLDER
    else:
        print(f"Unknown resolution {resolution}. Abort")
        return

    sst_filename, sst_filepath = find_file(input_folder, source_id)

    ds = xr.open_dataset(sst_filepath)

    # Squeeze out dimensions of size one, if any (e.g. member_id, dcpp_init_year)
    squeezed_ds = ds.squeeze(drop=True)

    # remove unnecessary coordinates
    squeezed_ds = squeezed_ds.reset_coords('month', drop=True)
    
    if source_id.split("-")[0] != "HADISST" and source_id.split("-")[0] != "GODAS":
        squeezed_ds = squeezed_ds.reset_coords('areacello', drop=True)

    # select sst anomalies for the required time period
    sst_anom = squeezed_ds["tos_anom_1m"].sel(time=slice(start_date, end_date))

    # Change the "time" index to the 15th day of the resulting month at midnight to enable alignment with other datasets
    sst_anom["time"] = (
        sst_anom.indexes["time"]
        .to_series()
        .apply(lambda x: x.replace(day=15, hour=0, minute=0, second=0))
        .values
    )

    if lat_slice is not None:
        sst_anom = sst_anom.sel(lat=lat_slice)

    if lon_slice is not None:
        sst_anom = sst_anom.sel(lon=lon_slice)
    
    
    num_orig_samples = sst_anom.coords['time'].size
    
    # Use rolling window and stack to add input_time_step dimension
    sst_anom_rolling = (
        sst_anom
        .rolling(time=num_input_time_steps, center=False)
        .construct('input_time_step')
    )

     # Drop the NaN values that arise due to rolling window
    sst_anom_rolling = sst_anom_rolling.isel(time=slice(num_input_time_steps - 1, None))

    # Transpose to get the desired order of dimensions
    sst_anom = sst_anom_rolling.transpose('time', 'input_time_step', 'lat', 'lon')

    # Compute the number of output samples
    num_output_samples = sst_anom.sizes['time']
    
    # Returns the total number of elements in the input tensor.
    total_nb_entries = sst_anom.size
    
    # count nan values
    nb_nan_entries = np.count_nonzero(np.isnan(sst_anom))
    

    if fill_nan: 
        sst_anom = sst_anom.fillna(0)
        
        

    if data_format == "flatten":
        # result is a numpy ndarray
        # sst_anom = flatten_lonlat(sst_anom, verbose=True)
        sst_anom = flatten_lonlat_and_window(sst_anom, verbose=False)
        
        if use_pca:
            pca = PCA(n_components=pca_components)
            pca.fit(sst_anom)
            sst_anom = pca.transform(sst_anom)
        
    nb_nan_entries_flat = np.count_nonzero(np.isnan(sst_anom))
    
    # if remove_win: 
    #     sst_anom = remove_window_dimension(sst_anom)

    if verbose:
        print(f"\nPredictors:")
        print(f"• Search for `{source_id}` SST file in {input_folder}")
        print(f"• Load {sst_filename}", end=" ")
        print(f"({ds['time'][0].values.astype('datetime64[D]')} to {ds['time'][-1].values.astype('datetime64[D]')})")
        print(f"• Slice the time period from {start_date.date()} to {end_date.date()}")

        print(f"• Nb of original samples (nb months): {num_orig_samples}")
        # print(f"• Stack {num_input_time_steps} previous time steps for each sample")
        print(f"• Nb of output samples, each containing {num_input_time_steps} time steps: {num_output_samples}")
        if data_format == "flatten":
            print(
                f"• Reshape sst anomalies into a 2D {sst_anom.shape} array: {num_output_samples} months x {sst_anom.shape[1]:,} num_input_time_steps*lon*lat values per month")
        else:
            print(
                f"• sst anomalies are a 4D {sst_anom.shape} array: {num_output_samples} months x {sst_anom.shape[1]:,} num_input_time_steps, {sst_anom.shape[2]:,} lat, {sst_anom.shape[3]:,} lon values per month")
        if fill_nan:
            print(f"• Replace {nb_nan_entries:,} nan values (continents) out of {total_nb_entries:,} with 0's ({nb_nan_entries/total_nb_entries:.2%})")
        else: 
            print(f"• {nb_nan_entries:,} nan values (continents) out of {total_nb_entries:,} ({nb_nan_entries / total_nb_entries:.2%}) - no replacement")
            
        if data_format == "flatten":
            print(
                f"• The dimensions are: {sst_anom.shape}: {sst_anom.shape[0]} months, {sst_anom.shape[1]:,} num_input_time_steps*lon*lat values per month")
        else:
            print(
                f"• The dimensions are: {sst_anom.shape}: {num_output_samples} months x {sst_anom.shape[1]:,} num_input_time_steps, {sst_anom.shape[2]:,} lat, {sst_anom.shape[3]:,} lon values per month")

    ds.close()
    squeezed_ds.close()

    #### TARGET VALUES ####
    if source_id.split("-")[0] == "HADISST":
        if target == "oni":
            target_folder = HADISST_ONI_FOLDER
            target_var_name = 'oni'
        elif target == "nino34":
            target_folder = HADISST_NINO34_FOLDER
            target_var_name = 'nino34'
        elif target == "E":
            target_folder = HADISST_EC_FOLDER
            target_var_name = 'E'
        elif target == "C":
            target_folder = HADISST_EC_FOLDER
            target_var_name = 'C'
        else:
            print(f"Unknown target {target}. Abort")
            return
    elif source_id.split("-")[0] == "GODAS":
        if target == "oni":
            target_folder = GODAS_ONI_FOLDER
            target_var_name = 'oni'
        elif target == "nino34":
            target_folder = GODAS_NINO34_FOLDER
            target_var_name = 'nino34'
        elif target == "E":
            target_folder = GODAS_EC_FOLDER
            target_var_name = 'E'
        elif target == "C":
            target_folder = GODAS_EC_FOLDER
            target_var_name = 'C'
        else:
            print(f"Unknown target {target}. Abort")
            return
    else: 
        if target == "oni":
            target_folder = CMIP6_ONI_FOLDER
            target_var_name = 'oni'
        elif target == "nino34":
            target_folder = CMIP6_NINO34_FOLDER
            target_var_name = 'nino34'
        elif target == "E":
            target_folder = CMIP6_EC_FOLDER
            target_var_name = 'E'
        elif target == "C":
            target_folder = CMIP6_EC_FOLDER
            target_var_name = 'C'
        else:
            print(f"Unknown target {target}. Abort")
            return

    target_filename, target_filepath = find_file(target_folder, source_id)

    df = pd.read_csv(target_filepath, sep='\t', index_col=0)
    df.index = pd.to_datetime(df.index)

    target_ts = df[target_var_name]

    lead_offset = pd.DateOffset(months=lead_time)
    num_input_time_steps_offset = pd.DateOffset(months=num_input_time_steps - 1)
    # num_input_time_steps_offset = pd.DateOffset(months=0) 
    start_date_plus_lead = start_date + lead_offset + num_input_time_steps_offset
    # print(f"start_date_plus_lead: {start_date_plus_lead}")
    end_date_plus_lead = end_date + lead_offset

    target_ts = target_ts[start_date_plus_lead:end_date_plus_lead]

    if verbose:
        print(f"\nPredictands:")
        print(f"• Search for `{source_id}` target file in {target_folder}")
        print(f"• Load {target_filename}", end=" ")
        print(f"({df.index[0].date()} to {df.index[-1].date()})")
        print(
            f"• Slice the time period with {lead_time:02d} month(s) lead time and {num_input_time_steps_offset.months} months offset: from {start_date_plus_lead.date()} to {end_date_plus_lead.date()}")
        print(f"• The dimensions are: {target_ts.shape}: {target_ts.shape[0]} months, 1 target value per month\n")

    # return sst_anom, target_ts
    return sst_anom, target_ts.astype(np.float32)


def check_matrix_properties(M):
        np.set_printoptions(precision = 2, suppress = True)
        print("\nMatrix Properties:")
        print("• Valid lagged correlation matrix's shape: ", M.shape)
        print(f"• Covariance matrix min and max values: {M.min():>5.2f}, {M.max():>5.2f}")
        
        # Check symmetry
        is_symmetric = np.allclose(M, M.T)
        
        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvals(M)
        is_positive_semi_definite = np.all(eigenvalues >= 0)
                
        print("• Is Matrix Symmetric? ", is_symmetric)
        print("• Is Matrix Positive Semi-Definite? ", is_positive_semi_definite)
        print("• Eigenvalues: ", eigenvalues)
        
    
        has_nans = np.isnan(M).any()
        print("• Does Matrix contains NaNs:", has_nans)
    
        # Compute the variance for each feature
        variances = np.var(M, axis=0)
        
        # Identify zero variance features
        zero_variance_mask = variances == 0
        zero_variance_indices = np.where(zero_variance_mask)[0]

        print("• Variances of features: ", variances)
        print("• Zero variance features at indices:", zero_variance_indices)
        print()
        

def compute_adj_matrix(corr_matrix, corr_coef_thresh):
    # If corr. Coef. of pair > threshold, then connected (Aij =1). Else, not connected (Aij =0).
    adj_matrix = np.where(corr_matrix > corr_coef_thresh, 1, 0)
    return adj_matrix


def flatten_lonlat(data: xr.DataArray, verbose=False):

    # Stack lat and lon dimensions into a single lonlat new dimension.
    flattened_data = data.stack(points=('lon', 'lat'))

    # Convert to a numpy array
    flattened_data_np = flattened_data.values
    
    if verbose: 
        print("Flatten DataArray lat and lon dimensions:")
        print("• Original array's shape (time, window, lat, lon)", data.shape)
        print("• Flattend array's shape (time, window, lonlat)", flattened_data_np.shape)

    
    return flattened_data_np

def flatten_lonlat_and_window(data: xr.DataArray, verbose=False):

    # Stack input_time_step, lat, and lon dimensions into a single points_with_win dimension.
    flattened_data = data.stack(points_with_win=('input_time_step', 'lon', 'lat'))
    
    # Convert to a numpy array
    flattened_data_np = flattened_data.values
    
    if verbose: 
        print("Flatten DataArray input_time_step, lat, and lon dimensions:")
        print("• Original array's shape (time, window, lat, lon)", data.shape)
        print("• Flattened array's shape (time, points_with_win)", flattened_data_np.shape)

    
    return flattened_data_np



def plot_weights_bars(weights, vmax=False, title=False):
    # inspired by CCAI tutorial
    plt.figure(figsize=(15, 1.5))
    

    plt.bar(range(len(weights)), weights, label='Weights')
    plt.ylabel('Learned Weight Value', fontsize=9)
    
    if vmax:
        plt.ylim((None, vmax))
        
    if title:
        plt.title(f'{title}', fontsize=10)
    else: 
        plt.title(f'Size of Weights \n(number of weights: {len(weights)})', fontsize=10)

    plt.grid(linestyle='--', linewidth=0.5)
    plt.show()
    
    
def plot_weights_bars_pca(model_results, target, lead_time, pca_components=False, vmax=False, subtitle=False, save_img=False, img_filename=None, img_folder=None):
    # inspired by CCAI tutorial
    
    # Get the set of results corresponding to 1 month lead time
    model_dict = model_results[target][lead_time]
    weights = model_dict['weights']
    print("weights.shape: ", weights.shape)
        
    subplot_width = 5
    subplot_height = 4
    fig, ax = plt.subplots(figsize=(subplot_width, subplot_height), nrows=1, ncols=1, sharey=True, layout='compressed')
    
    ax.bar(range(len(weights)), weights, label='Weights')
    # ax.set_title(f"input time step = {i}", fontsize=9)
    ax.set_xlabel(f"Weight number")
    
    ax.set_ylabel(f"Learned Weight Value", fontsize=9)
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    if vmax:
        ax.set_ylim((-vmax, vmax))
        
    fig.suptitle(f"Size of {model_dict['model_name']} Learned Weights for predicting the {target.capitalize()} index with a lead time of {lead_time}\n"
                 f"Number of weights per input time step: {weights.size} (PCA components)\n\n"
                 f"{subtitle}",
                fontsize=10, fontweight="bold")

    if save_img == True:
        if img_filename == None:
            img_filename = f"coef_bar_plot_pca_{model_dict['experiment_name']}.jpg"
        fig.savefig(img_folder+img_filename, dpi=300, bbox_inches='tight')
    
    plt.show()
            
    if save_img == True:
        print(f"Saved {img_filename} to disk")
            
            
def plot_weights_bars_v2(model_results, target, lead_time, input_time_steps, nb_lon_gridpoints, nb_lat_gridpoints, lonlat_coord_pairs, invalid_indices, vmax=False, subtitle=False, save_img=False, img_filename=None, img_folder=None):
    # inspired by CCAI tutorial
    
    # Get the set of results corresponding to 1 month lead time
    model_dict = model_results[target][lead_time]
    weights = model_dict['weights']
    print("weights.shape: ", weights.shape)
           
    subplot_width = 25/input_time_steps
    subplot_height = 8
    fig, axs = plt.subplots(figsize=(subplot_width, subplot_height), nrows=1, ncols=input_time_steps, sharey=True, layout='compressed')
    

    # Reshape the weights back to the original dimensions
    weights_3d =  model_dict['weights'].reshape((input_time_steps, nb_lon_gridpoints, nb_lat_gridpoints))
    print("weights_3d.shape: ", weights_3d.shape)


    # label every "interval" tick
    ticks = range(0, len(lonlat_coord_pairs), nb_lon_gridpoints*4)  
    secondary_labels = [lonlat_coord_pairs[i] for i in ticks]  

    
    
    for i in range(input_time_steps):
        if input_time_steps == 1:
            ax = axs
        else:
            ax = axs[i]


        # Display the weights as a 2D image
        weights_2d = weights_3d[i, :, :]
        weights_1d = weights_2d.flatten()
        ax.bar(range(len(weights_1d)), weights_1d, label='Weights')
        ax.set_title(f"input time step = {i}", fontsize=9)
        ax.set_xlabel(f"Weight number")
        ax.set_xticks(ticks)
        
        if i == 0:
            ax.set_ylabel(f"Learned Weight Value", fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.5)
        
        if vmax:
            ax.set_ylim((-vmax, vmax))
        
        # Create a secondary x-axis
        ax2 = ax.twiny()  # Create a secondary x-axis that shares the y-axis
        ax2.set_xlim(ax.get_xlim())  # Set the limits to match the main x-axis

        ax2.set_xticks(ticks)
        ax2.set_xticklabels(secondary_labels)
        ax2.tick_params(axis='x', labelsize=7, rotation=45, direction='out', pad=5)


        # Adjust the position of the secondary x-axis labels
        ax2.xaxis.set_ticks_position('top')  # Set the ticks position to top
        ax2.xaxis.set_label_position('top')  # Set the label position to top
        ax2.spines['top'].set_position(('outward', 0))  # Reduce the outward position
        
        # put a grey vertical line at continental indices
        # ax.bar(invalid_indices, 0.1, label='Weights')
        
        for invalid_index in invalid_indices:
            if invalid_index == invalid_indices[0]:
                ax.axvline(x=invalid_index, color='grey', linewidth=0.5, alpha=0.1, label='Continents')
            else:
                ax.axvline(x=invalid_index, color='grey', linewidth=0.5, alpha=0.1)
        
        ax.legend(loc='upper right')
            
    fig.suptitle(f"Size of Learned Weights for predicting the {target.capitalize()} index with a lead time of {lead_time} (corr={model_dict['corr']:.2f}, mse={model_dict['mse']:.2f})\n"
                 f"Number of weights {weights.size} ({weights_3d.shape[0]}x{weights_3d.shape[1]}x{weights_3d.shape[2]})\n\n"
                 f"{subtitle}",
                fontsize=10, fontweight="bold")
    
    if save_img == True:
        if img_filename == None:
            img_filename = f"coef_bar_plot_{model_dict['experiment_name']}.jpg"
        fig.savefig(img_folder+img_filename, dpi=300, bbox_inches='tight')
        
    plt.show()

    if save_img == True:
        print(f"Saved {img_filename} to disk")
        
    
    
def plot_weights_on_grid(model_results, target, lead_time, input_time_steps, min_lon, max_lon, min_lat, max_lat, resolution, nb_lon_gridpoints, nb_lat_gridpoints, subtitle=None, save_img=False, img_filename=None, img_folder=None):
    
    lon_ticks = np.arange(0, nb_lon_gridpoints, 1)
    lon_labels = np.arange(min_lon, max_lon, resolution)
    lat_ticks = np.arange(0, nb_lat_gridpoints, 1)
    lat_labels = np.arange(min_lat, max_lat, resolution)
    
    lon_tick_labels_step = 60
    lat_tick_labels_step = 25
    
    # adjust to resolution
    lon_tick_labels_step //= resolution
    lat_tick_labels_step //= resolution

    
    # Get the set of results corresponding to 1 month lead time
    model_dict = model_results[target][lead_time]
    weights = model_dict['weights']
    print("weights.shape: ", weights.shape)
    
    
    subplot_width = 5*input_time_steps
    subplot_height = 3
    fig, axs = plt.subplots(figsize=(subplot_width, subplot_height), nrows=1, ncols=input_time_steps, sharey=True, layout='compressed')
    
    
    # Since we want the colorbar to be symetrical around zero, we need to take the weights' maximum abs value and use it for vmin and vmax
    w_min, w_max = np.inf, -np.inf
    w_min = min(w_min, weights.min())
    w_max = max(w_max, weights.max())
    # print(f"min and max coefficients: {w_min:.4f}, {w_max:.4f}")
    abs_w_max = max(abs(w_min), abs(w_max))
    
    # Reshape the weights back to the original dimensions
    weights_3d =  model_dict['weights'].reshape((input_time_steps, nb_lon_gridpoints, nb_lat_gridpoints))

    # Reshape weights into a 2D grid corresponding to latitude and longitude dimensions        
    # weights_2d = model_dict['weights'].reshape((360, 180))
    # weights_2d = model_dict['weights'].reshape((nb_lon_gridpoints, nb_lat_gridpoints))
    # print("weights_2d.shape: ", weights_2d.shape)

    # Calculate the average weights across input_time_steps
    # weights_2d = np.mean(weights_3d, axis=0)

    for i in range(input_time_steps):
        if input_time_steps == 1:
            ax = axs
        else:
            ax = axs[i]


        # Display the weights as a 2D image
        weights_2d = weights_3d[i, :, :]
        cax = ax.imshow(weights_2d.T, cmap='bwr', vmin=-abs_w_max, vmax=abs_w_max, origin='lower')
        # ax.set_title(f"lead time = {model_dict['lead_time']} month(s)", fontsize=9)
        ax.set_title(f"input time step = {i}", fontsize=9)
        ax.set_xlabel(f"Longitude (deg east)")
        if i == 0:
            ax.set_ylabel(f"Latitude (deg)")
    
            
        # set the tick positions and labels at specified intervals
        # Apply the slicing operation (collection[start:stop:step])
        ax.set_xticks(lon_ticks[::lon_tick_labels_step])
        ax.set_xticklabels(lon_labels[::lon_tick_labels_step])
        ax.set_yticks(lat_ticks[::lat_tick_labels_step])
        ax.set_yticklabels(lat_labels[::lat_tick_labels_step])

    # Create colorbar
    if input_time_steps == 1:
        cb = plt.colorbar(cax, ax=ax, orientation='vertical')
    else: 
        cb = plt.colorbar(cax, ax=axs[input_time_steps-1], orientation='vertical')
    cb.set_label(f'Model coefs')
           
    fig.suptitle(f"Coefficients for predicting the {target} index with lead time of {lead_time} (corr={model_dict['corr']:.2f}, mse={model_dict['mse']:.2f})\n"
                 f"Number of weights {weights.size} ({weights_3d.shape[0]}x{weights_3d.shape[1]}x{weights_3d.shape[2]})\n\n"
                 f"{subtitle}",
                #  f"Trained on on {model_dict['train_source_ids']} data ({utils.add_cardinals_fname(min_lat, max_lat, min_lon, max_lon)}) from {model_dict['train_start_date']} to {model_dict['train_end_date']}",
                fontsize=10, fontweight="bold")

    if save_img == True:
        if img_filename == None:
            img_filename = f"coef_grid_plot_{model_dict['experiment_name']}.jpg"
        fig.savefig(img_folder+img_filename, dpi=300, bbox_inches='tight')
        
    plt.show()
    
    if save_img == True:
        print(f"Saved {img_filename} to disk")


def plot_gradients_on_grid_all_leads(gnn_gradients_results_target, property, model_results, target, input_time_steps, min_lon, max_lon, min_lat, max_lat, resolution, 
                                     nb_lon_gridpoints, nb_lat_gridpoints, subtitle=None, save_img=False, img_filename=None, img_folder=None):
    max_lead_time = len(gnn_gradients_results_target)
    
    if property == "avg_gradient_per_node_fullmap":
      subtitle = f"Average gradient per node for 1 Epoch \n{subtitle}"    
    elif property == "last_graph_gradients_fullmap":
      subtitle = f"Last graph gradients of the epoch \n{subtitle}"    
    else:
        print("Unknown property. Abort")
        return


    subplot_width = 40/input_time_steps
    subplot_height = 2 * max_lead_time
    fig, axs = plt.subplots(figsize=(subplot_width, subplot_height), nrows=max_lead_time, ncols=input_time_steps, sharey=True, layout='compressed')
        
        
    for lead_time, ax_row in zip(range(max_lead_time), axs):     
        gradients = gnn_gradients_results_target[lead_time][property]
            
        model_dict = model_results[target][lead_time]

        lon_ticks = np.arange(0, nb_lon_gridpoints, 1)
        lon_labels = np.arange(min_lon, max_lon, resolution)
        lat_ticks = np.arange(0, nb_lat_gridpoints, 1)
        lat_labels = np.arange(min_lat, max_lat, resolution)
        
        lon_tick_labels_step = 60
        lat_tick_labels_step = 25
        
        # adjust to resolution
        lon_tick_labels_step //= resolution
        lat_tick_labels_step //= resolution

        # Get the set of results corresponding to 1 month lead time
        # print("gradients.shape: ", gradients.shape)
                
        # Since we want the colorbar to be symetrical around zero, we need to take the gradients' maximum abs value and use it for vmin and vmax
        w_min, w_max = np.inf, -np.inf
        w_min = min(w_min, gradients.min())
        w_max = max(w_max, gradients.max())
        # print(f"min and max coefficients: {w_min:.4f}, {w_max:.4f}")
        abs_w_max = max(abs(w_min), abs(w_max))
        
        
        
        # Reshape the weights back to the original dimensions
        gradients_3d =  gradients.reshape((nb_lon_gridpoints, nb_lat_gridpoints, input_time_steps))
        # print("gradients_3d.shape: ", gradients_3d.shape)
        # Calculate the average weights across input_time_steps
        # weights_2d = np.mean(weights_3d, axis=0)

        for i, ax in zip(range(input_time_steps), ax_row):

            # Display the gradients as a 2D image
            gradients_2d = gradients_3d[:, :, i]
            cax = ax.imshow(gradients_2d.T, cmap='bwr', vmin=-abs_w_max, vmax=abs_w_max, origin='lower')
            # ax.set_title(f"lead time = {model_dict['lead_time']} month(s)", fontsize=9)
            
            title = (
                f"Lead Time: {lead_time} Months - Corr: {model_dict['corr']:.2f}, MSE: {model_dict['mse']:.2f}\n"
                f"input time step = {i}")
             
            # ax.set_title(f"input time step = {i}", fontsize=9)
            ax.set_title(f"{title}", fontsize=9)
            ax.set_xlabel(f"Longitude (deg east)")
            if i == 0:
                ax.set_ylabel(f"Latitude (deg)")
        
                
            # set the tick positions and labels at specified intervals
            # Apply the slicing operation (collection[start:stop:step])
            ax.set_xticks(lon_ticks[::lon_tick_labels_step])
            ax.set_xticklabels(lon_labels[::lon_tick_labels_step])
            ax.set_yticks(lat_ticks[::lat_tick_labels_step])
            ax.set_yticklabels(lat_labels[::lat_tick_labels_step])

        # # Create colorbar
        cbar = fig.colorbar(cax, ax=ax_row, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Gradients')

            
        fig.suptitle(f"Gradient-Based Attribution for Predicting the {target.capitalize()} Index\n\n"
                    f"{subtitle}",
                    fontsize=10, fontweight="bold")

    if save_img == True:
        if img_filename == None:
            img_filename = f"gradients_grid_plot_{model_dict['experiment_name']}.jpg"
        fig.savefig(img_folder+img_filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    if save_img == True:
        print(f"Saved {img_filename} to disk")

import matplotlib.ticker as ticker

def plot_gradients_on_grid(gradients, model_results, target, lead_time, input_time_steps, min_lon, max_lon, min_lat, max_lat, resolution, nb_lon_gridpoints, nb_lat_gridpoints, 
                           abs_gradient_max=None, last_time_step_only=False, subtitle=None, save_img=False, img_filename=None, img_folder=None, display=True):
    model_dict = model_results[target][lead_time]

    lon_ticks = np.arange(0, nb_lon_gridpoints, 1)
    lon_labels = np.arange(min_lon, max_lon, resolution)
    lat_ticks = np.arange(0, nb_lat_gridpoints, 1)
    lat_labels = np.arange(min_lat, max_lat, resolution)
    
    lon_tick_labels_step = 60
    lat_tick_labels_step = 25
    
    # adjust to resolution
    lon_tick_labels_step //= resolution
    lat_tick_labels_step //= resolution


    if last_time_step_only:
        ncols= 1
    else:
        ncols=input_time_steps

    
    # Get the set of results corresponding to 1 month lead time
    # print("gradients.shape: ", gradients.shape)    
    
    
    subplot_width = 10
    subplot_height = 3
    fig, axs = plt.subplots(figsize=(subplot_width, subplot_height), nrows=1, ncols=ncols, sharey=True, layout='compressed')
    
    if abs_gradient_max: 
        abs_w_max = abs_gradient_max
    else:    
        # Since we want the colorbar to be symetrical around zero, we need to take the gradients' maximum abs value and use it for vmin and vmax
        w_min, w_max = np.inf, -np.inf
        w_min = min(w_min, gradients.min())
        w_max = max(w_max, gradients.max())
        # print(f"min and max coefficients: {w_min:.4f}, {w_max:.4f}")
        abs_w_max = max(abs(w_min), abs(w_max))
    
    
    
    # Reshape the weights back to the original dimensions
    gradients_3d =  gradients.reshape((nb_lon_gridpoints, nb_lat_gridpoints, input_time_steps))
    # print("gradients_3d.shape: ", gradients_3d.shape)
    # Calculate the average weights across input_time_steps
    # weights_2d = np.mean(weights_3d, axis=0)

    for i in range(input_time_steps):
        if last_time_step_only and i != input_time_steps-1:
            continue

        if ncols == 1:
            ax = axs
        else:
            ax = axs[i]

        # Display the gradients as a 2D image
        gradients_2d = gradients_3d[:, :, i]
        cax = ax.imshow(gradients_2d.T, cmap='bwr', vmin=-abs_w_max, vmax=abs_w_max, origin='lower')
        ax.set_title(f"lead time = {model_dict['lead_time']} month(s)", fontsize=9)
        # ax.set_title(f"input time step = {i}", fontsize=9)
        # ax.set_title(f"input time step = {i}", fontsize=9)
        ax.set_xlabel(f"Longitude (deg east)")
        # if i == 0:
        ax.set_ylabel(f"Latitude (deg)")
    
            
        # set the tick positions and labels at specified intervals
        # Apply the slicing operation (collection[start:stop:step])
        ax.set_xticks(lon_ticks[::lon_tick_labels_step])
        ax.set_xticklabels(lon_labels[::lon_tick_labels_step])
        ax.set_yticks(lat_ticks[::lat_tick_labels_step])
        ax.set_yticklabels(lat_labels[::lat_tick_labels_step])

    # Create colorbar
    if ncols == 1:
        cb = plt.colorbar(cax, ax=ax, orientation='vertical', pad=0.01)
    else: 
        cb = plt.colorbar(cax, ax=axs[input_time_steps-1], orientation='vertical', pad=0.01)

    # Force color bar to always use scientific notation
    cb.formatter = ticker.FuncFormatter(lambda x, _: f"{x:.0e}")
    cb.update_ticks()

    # cb.set_label(f'Gradients')    
    
    if subtitle is not None:
        fig.suptitle(f"Gradient-Based Attribution for Predicting the {target.capitalize()} Index with a Lead Time of {lead_time} Months (corr={model_dict['corr']:.2f}, mse={model_dict['mse']:.2f})\n\n"
                    #  f"Number of weights {weights.size} ({weights_3d.shape[0]}x{weights_3d.shape[1]}x{weights_3d.shape[2]})\n\n"
                    f"{subtitle}",
                    #  f"Trained on on {model_dict['train_source_ids']} data ({utils.add_cardinals_fname(min_lat, max_lat, min_lon, max_lon)}) from {model_dict['train_start_date']} to {model_dict['train_end_date']}",
                    fontsize=10, fontweight="bold")

    if save_img == True:
        if img_filename == None:
            img_filename = f"gradients_grid_plot_{model_dict['experiment_name']}.jpg"
        fig.savefig(img_folder+img_filename, dpi=300, bbox_inches='tight')
        
    if display == True:    
        plt.show()
    else:
        plt.close()
    
    if save_img == True:
        print(f"Saved {img_filename} to folder {img_folder}")
      


def handle_invalid_values(flatten_data: np.ndarray, nan_indices: list[int], method: str, verbose=False):

    if verbose:
        print("Handle invalid values:")    
        print("• Original data's shape (time, lonlat)", flatten_data.shape)    

    if method == "complete_case":
        if verbose:
            print(f"• Method='complete_case': Remove {nan_indices.size} invalid lonlat indices") 
        
        # Remove columns with NaN values
        processed_data = np.delete(flatten_data, nan_indices, axis=2)
            
    elif method == "mask":
        if verbose:
            print(f"• Method='mask': Mask {nan_indices.size} invalid lonlat indices") 

        # Create a mask with the same shape as the 2D array, initialized to False
        mask = np.zeros_like(flatten_data, dtype=bool)

        # set the mask for all time steps at the specified lonlat indices
        mask[:, :, nan_indices] = True   
    
        # Apply the mask to the data
        processed_data = ma.masked_array(flatten_data, mask=mask)
    else:
        raise ValueError("Invalid method. Use 'complete_case' or 'mask'.")

    if verbose:
        print(f"• Processed data's shape: {processed_data.shape}")
    return processed_data


def compute_corr_matrix(flattened_data: np.ndarray, masked_data=False):
    """
    Compute the Pearson correlation coefficient matrix
    rowvar = False --> each column represents a variable, while the rows contain observations.
    
    sst_anom: DataArray of dimensions (time, lonlat)
                                 e.g. (1440, 1656)
                                 
    """
    no_win_flattened_data = remove_window_dimension(flattened_data)

    if masked_data == True:
        # very slow !
        correlation_matrix = ma.corrcoef(no_win_flattened_data, allow_masked=True, rowvar=False)
    else:
        correlation_matrix = np.corrcoef(no_win_flattened_data, rowvar=False)

    return correlation_matrix


def compute_cross_corr_matrix(sst_data, max_lead_time):
    """
    Calculate cross-correlation matrices for each lead time.
    
    Parameters:
    sst_data (np.ndarray): 2D array of SST anomalies with shape (num_time_steps, lon*lat)
    max_lead_time (int): Maximum lead time to consider
    
    Returns:
    dict: dictionary where keys are lead times and values are cross-correlation matrices.
    """
    num_time_steps, num_grid_points = sst_data.shape
    cross_corr_matrices = {}
    
    for lead_time in range(max_lead_time + 1):
        # Extract time series with the given lead time
        series1 = sst_data[:num_time_steps-lead_time, :]
        series2 = sst_data[lead_time:, :]
        
        # Combine the two series to form a single matrix for np.corrcoef
        combined_series = np.concatenate((series1, series2), axis=1)
        
        # Calculate the correlation matrix
        corr_matrix = np.corrcoef(combined_series, rowvar=False)
        
        # Extract the cross-correlation matrix from the full correlation matrix (top right corner)
        cross_corr_matrix = corr_matrix[num_grid_points:, :num_grid_points]
        
        cross_corr_matrices[lead_time] = cross_corr_matrix
    
    return cross_corr_matrices


def compute_corr_matrix_per_lead(flattened_data: np.ndarray, lead_time):
    """
    Calculate correlation matrices for each lead time.
    
    Parameters:
    flattened_data (np.ndarray): 2D array of SST anomalies with shape (num_time_steps, lon*lat)
    lead_time (int): 
    
    Returns:
    cross-correlation matrix for that lead time
    """
    num_time_steps, num_grid_points = flattened_data.shape
    
    # Extract time series with the given lead time
    original_series = flattened_data[:num_time_steps-lead_time, :]
    shifted_series = flattened_data[lead_time:, :]
    
    # # Combine the two series to form a single matrix for np.corrcoef
    # combined_series = np.concatenate((original_series, shifted_series), axis=1)
    
    # # Calculate the correlation matrix
    # full_corr_matrix = np.corrcoef(combined_series, rowvar=False)


    full_corr_matrix = np.cov(original_series, shifted_series)
    
    # Extract the cross-correlation matrix from the full correlation matrix (top right and bottom left corners)
    upper_right_corr_matrix = full_corr_matrix[num_grid_points:, :num_grid_points]
    lower_left_corr_matrix = full_corr_matrix[:num_grid_points, num_grid_points:]

    # get the max correlation point wise (for ex, max(Cov(node1,node3+lag), Cov(node3,node1+lag))
    max_corr_matrix = np.maximum(lower_left_corr_matrix, upper_right_corr_matrix)
    
    return max_corr_matrix



def insert_valid_corr_into_full(corr_matrix_valid, invalid_indices, corr_matrix_full_order):
    # Create sets with all and non-valid indices
    all_indices = set(np.arange(corr_matrix_full_order))
    non_valid_indices = set(invalid_indices)

    # Subtract the sets to get the valid indices
    valid_indices = np.array(list(all_indices - non_valid_indices))

    # Create a full correlation matrix filled with 0s
    corr_matrix_full = np.zeros((corr_matrix_full_order, corr_matrix_full_order))
    
    # Insert clean correlation matrix back into full correlation matrix valid indices
    for i, idx_i in enumerate(valid_indices):
        for j, idx_j in enumerate(valid_indices):
            corr_matrix_full[idx_i, idx_j] = corr_matrix_valid[i, j]

    return corr_matrix_full


    
def remove_window_dimension(flattened_data: np.ndarray):
    return flattened_data.copy()[:,-1,:]
    
def compute_cov_matrix_sklearn(flattened_data: np.ndarray, assume_centered=False):
    """
    Compute the Maximum likelihood covariance estimator.
    assume_centered: bool, default=False. If True, data will not be centered before computation. Useful when working with data whose mean is almost, but not exactly zero. If False, data will be centered before computation. (sklearn doc)

    flattened_data: DataArray of dimensions (time, input_time_steps, lonlat)
                                 e.g. (1440, 3, 1656)
    """
    no_win_flattened_data = remove_window_dimension(flattened_data)
    covariance_matrix = empirical_covariance(no_win_flattened_data, assume_centered=assume_centered)
    return covariance_matrix


# Mathematically, this shrinkage consists in reducing the ratio between the smallest and the largest eigenvalues 
# of the empirical covariance matrix. It can be done by simply shifting every eigenvalue according to a given offset, 
# which is equivalent of finding the l2-penalized Maximum Likelihood Estimator of the covariance matrix (SckitLearn doc)
def make_psd(cov_matrix, epsilon=1e-10, shrinkage=None):
    # Add a small value to the diagonal elements to ensure the matrix is PSD
    # cov_matrix_psd = cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon

    # Set shrinkage closer to 1 for poorly-conditioned data
    # https://stats.stackexchange.com/questions/172911/graphical-lasso-numerical-problem-not-spd-matrix-result
    print(f"•  Make cov matrix PSD. Shrinkage={shrinkage}")
    shrunk_cov = covariance.shrunk_covariance(cov_matrix, shrinkage=shrinkage) 

    return shrunk_cov
    

def compute_lagged_cov_matrix(flattened_data: np.ndarray, lead_time: int, assume_centered=False, shrinkage=None):
    """
    Compute the Maximum likelihood covariance estimator.
    assume_centered: bool, default=False. If True, data will not be centered before computation. Useful when working with data whose mean is almost, but not exactly zero. If False, data will be centered before computation. (sklearn doc)

    flattened_data: DataArray of dimensions (time, lonlat)
                                 e.g. (1440, 1656)
    """
    num_time_steps, num_grid_points = flattened_data.shape

    original_series = flattened_data
    shifted_series = np.roll(original_series, shift=-lead_time, axis=0)

    
    # Compute the covariance matrix of the combined data. The resulting covariance matrix will be 
    # 2n×2n, capturing the variances and covariances among the original and lagged time series.
    # lagged_cov_matrix = empirical_covariance(combined_series, assume_centered=assume_centered)
    lagged_cov_matrix = np.cov(original_series, shifted_series, rowvar=False)    

    # if lead_time != 0:
    #     shrinkage= 2/lead_time
    #     shrinkage= (-lead_time / 10) + 1
    
    adjusted_cov_matrix = make_psd(lagged_cov_matrix, shrinkage=shrinkage)
    
    return adjusted_cov_matrix




def compute_LedoitWolf_cov_matrix(flattened_data: np.ndarray, lead_time: int, assume_centered=False, shrinkage=None):
    """
    Compute the Maximum likelihood covariance estimator.
    assume_centered: bool, default=False. If True, data will not be centered before computation. Useful when working with data whose mean is almost, but not exactly zero. If False, data will be centered before computation. (sklearn doc)

    flattened_data: DataArray of dimensions (time, lonlat)
                                 e.g. (1440, 1656)
    """
    num_time_steps, num_grid_points = flattened_data.shape

    original_series = flattened_data
    shifted_series = np.roll(original_series, shift=-lead_time, axis=0)

    
    # Compute the covariance matrix of the combined data. The resulting covariance matrix will be 
    # 2n×2n, capturing the variances and covariances among the original and lagged time series.
    # lagged_cov_matrix = empirical_covariance(combined_series, assume_centered=assume_centered)
    
    combined_series = np.hstack((original_series, shifted_series))
    
    # Fit the Ledoit-Wolf estimator
    LedoitWolf_model = LedoitWolf().fit(combined_series)

    if shrinkage is not None:
        # Access the estimated covariance matrix
        cov_matrix_psd = make_psd(LedoitWolf_model.covariance_, shrinkage=shrinkage)

    else:
        cov_matrix_psd = None
    
    return cov_matrix_psd, LedoitWolf_model.covariance_



def compute_edge_list(adj_matrix):
    # convert the dense matrix to a Compressed Sparse Row matrix 
    # (more efficient in terms of memory and computation)
    sparse_matrix = csr_matrix(adj_matrix)

    # Get the indices of the elements that are non-zero using numpy.nonzero()
    rows, cols = np.nonzero(sparse_matrix)

    # Combine the row and column indices to get the list of edges
    # edge_list = list(zip(rows, cols))

    # Filter to avoid self loops and duplicates for undirected graphs
    edge_list = [(i, j) for i, j in zip(rows, cols) if i < j]

    return edge_list


def compute_sparsity(M):
    return 1 - (np.count_nonzero(M) / M.size)


# def construct_graph_nodes(anom_dataArray: xr.DataArray):
#     """ 
#     lat_coords: array of lat coordinates
#     lon_coords: array of lon coordinates
#     lonlat_values: 2d array with value of variable for each coordinate

#     - with each node identified by its coordinates (lat lon)

#     Example: 
#     ```
#     node_id: 0, pos=(lon=2.5, lat=-52.5), node_label=(2.5, -52.5), sst_value: 0.02464096061885357
#     node_id: 1, pos=(lon=7.5, lat=-52.5), node_label=(7.5, -52.5), sst_value: 0.10649676620960236
#     node_id: 2, pos=(lon=12.5, lat=-52.5), node_label=(12.5, -52.5), sst_value: 0.20234745740890503
#     ```
#     """

#     lat_dim = anom_dataArray.coords['lat'].values  # array of lat coordinates
#     lon_dim = anom_dataArray.coords['lon'].values  # array of lon coordinates

#     # Create an empty graph
#     G = nx.Graph()

#     # add nodes
#     node_id = 0
#     for i, lat in enumerate(lat_dim):
#         for j, lon in enumerate(lon_dim):
#             node_label = f"({lon}, {lat})"

#             # print a few examples nodes at the beginning and end
#             if (lon < 17.5 and lat == -52.5) or (lon > 342.5 and lat == 57.5):
#                 print(f"node_id: {node_id}, pos=(lon={lon}, lat={lat}), node_label={node_label}")
#             if (lon == 17.5 and lat == -52.5):
#                 print(f"(...)")

#             # Add a single node and update its attributes
#             G.add_node(node_id, pos=(lon, lat), label=node_label)
#             node_id += 1
#     return G


def construct_graph_nodes(anom_dataArray: xr.DataArray):
    """ 
    lat_coords: array of lat coordinates
    lon_coords: array of lon coordinates
    lonlat_values: 2d array with value of variable for each coordinate

    - with each node identified by its coordinates (lat lon)

    Example: 
    ```
    node_id: 0, pos=(lon=2.5, lat=-52.5), node_label=(2.5, -52.5), sst_value: 0.02464096061885357
    node_id: 1, pos=(lon=7.5, lat=-52.5), node_label=(7.5, -52.5), sst_value: 0.10649676620960236
    node_id: 2, pos=(lon=12.5, lat=-52.5), node_label=(12.5, -52.5), sst_value: 0.20234745740890503
    ```
    """

    lat_dim = anom_dataArray.coords['lat'].values  # array of lat coordinates
    lon_dim = anom_dataArray.coords['lon'].values  # array of lon coordinates

    # Create an empty graph
    G = nx.Graph()

    # add nodes
    node_id = 0
    for _, lon in enumerate(lon_dim):
        for _, lat in enumerate(lat_dim):
            node_label = f"({lon}, {lat})"

            # print a few examples nodes at the beginning and end
            if (lon < 17.5 and lat == -52.5) or (lon > 342.5 and lat == 57.5):
                print(f"node_id: {node_id}, pos=(lon={lon}, lat={lat}), node_label={node_label}")
            if (lon == 17.5 and lat == -52.5):
                print(f"(...)")

            # Add a single node and update its attributes
            G.add_node(node_id, pos=(lon, lat), label=node_label)
            node_id += 1
    return G


def construct_graph_nodes_and_edges(anom_dataArray, edge_list):
    G = construct_graph_nodes(anom_dataArray)
    G.add_edges_from(edge_list)
    return G


# this version is compatible with PyG
def construct_onehop_edges(G, res):
    pos = nx.get_node_attributes(G, 'pos')
    min_lon, max_lon, min_lat, max_lat = get_min_lon_lat(G)

    reverse_pos_dict = get_reverse_pos_dict(G)

    for node, position in pos.items():
        lon = position[0]
        lat = position[1]
        if node % 10000 == 0:
            print(f"\nNode {node}: Position {position}")            
        for x in np.arange(lon-res, lon+res+res, res):
            for y in np.arange(lat-res, lat+res+res, res):

                # exclude neighbors out of bounds
                if x < min_lon or x > max_lon or y < min_lat or y > max_lat:
                    # print(f"({x}, {y})")
                    continue

                # exclude self loop
                if not (x, y) == (lon, lat):
                    dest = reverse_pos_dict[(x, y)]
                    if node % 10000 == 0:
                        print(f"Adding edge from Node {node} {pos[node]} to Node {dest} {pos[dest]}")
                    G.add_edge(node, dest)
    return G


def construct_onehop_edges_with_lon_wrapping(G, res):
    pos = nx.get_node_attributes(G, 'pos')
    min_lon, max_lon, min_lat, max_lat = get_min_lon_lat(G)

    print(f"min_lon: {min_lon}, max_lon: {max_lon}, min_lat: {min_lat}, max_lat: {max_lat}")
    reverse_pos_dict = get_reverse_pos_dict(G)

    for node, position in pos.items():
        lon = position[0]
        lat = position[1]
        if node % 10000 == 0:
            print(f"\nNode {node}: Position {position}")            
        for x in np.arange(lon-res, lon+res+res, res):
            for y in np.arange(lat-res, lat+res+res, res):
                
                # Wrap around the longitudinal edges
                if x < min_lon:
                    x = max_lon
                elif x > max_lon:
                    x = min_lon
                else:
                    pass # x is within bounds

                # exclude neighbors out of lateral bounds
                if y < min_lat or y > max_lat:
                    # print(f"({x}, {y})")
                    continue

                # add edge to graph (excluding self loops)
                if not (x, y) == (lon, lat):
                    dest = reverse_pos_dict[(x, y)]
                    if node % 10000 == 0:
                        print(f"Adding edge from Node {node} {pos[node]} to Node {dest} {pos[dest]}")
                    G.add_edge(node, dest)
    return G


def display_adj_matrix(adj_matrix, title):
    # Define a colormap with two colors
    cmap = ListedColormap(['white', 'black'])
    bounds = [0, 0.5, 1]
    norm = BoundaryNorm(bounds, cmap.N)

    # Plot the matrix
    plt.imshow(adj_matrix, cmap=cmap, norm=norm, interpolation='nearest')

    # Create a color bar with discrete values
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['0', '1'])  # set the labels to '0' and '1'

    plt.title(title, fontsize=10)
    plt.show()


def display_adj_matrix_w_coords(G, adj_matrix, full_pos_dict, title=None, save_img=False, img_filename=None, img_folder=None):
    # pos = nx.get_node_attributes(G, 'pos')

    # Define a colormap with two colors
    cmap = ListedColormap(['white', 'black'])
    bounds = [0, 0.5, 1]
    norm = BoundaryNorm(bounds, cmap.N)

    # Plot the matrix
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(8,8))
    cax = ax.imshow(adj_matrix, cmap=cmap, norm=norm, interpolation='nearest')

    # label every 200th tick
    ticks = range(0, adj_matrix.shape[0], 230)  
    labels = [full_pos_dict[i] for i in ticks]  

    # Set tick labels
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=6)

    ax.set_xlabel("Coordinates (lon, lat)")
    ax.set_ylabel("Coordinates (lon, lat)")

    # Create a color bar with discrete values
    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0', '1'])  # set the labels to '0' and '1'

    plt.grid(linestyle='--', linewidth=0.5)

    sparsity = compute_sparsity(adj_matrix)
    if title is not None:
        plt.title(title, fontsize=10)
    else:
        plt.title("Adjacency matrix", fontsize=10)

    if save_img == True:
        if img_filename is None:
            print("Please provide a filename to save the image to disk")
        else:
            fig.savefig(img_folder + img_filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    if save_img == True and img_filename is not None:
        print(f"Saved {img_filename} to disk")


def display_corr_matrix(corr_matrix, title=None, save_img=False, img_filename=None, img_folder=None):
    # cmap = plt.get_cmap('coolwarm')

    fig, ax = plt.subplots()

    # same cmap as plot_neighbors_corr
    levels = np.linspace(-1, 1, 21)
    cmap = plt.get_cmap('RdBu_r')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    #  Set masked or NaN values to be grey
    # masked_corr_matrix = ma.masked_invalid(corr_matrix)
    # cmap.set_bad(color='grey')  # Set masked or NaN values to be grey

    # Plot the correlation matrix
    im = ax.imshow(corr_matrix, cmap=cmap, norm=norm, interpolation='nearest')

    # Create a ScalarMappable for the colorbar
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Add the colorbar to the subplot
    cbar = fig.colorbar(smap, ax=ax)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    # cbar.ax.set_title('Correlation')

    if title is not None:
        plt.title(title, fontsize=10)
    else:
        plt.title(f"Correlation Matrix\n(Pairwise Node Correlation at Lag 0)", fontsize=10)

    if save_img == True:
        if img_filename is None:
            print("Please provide a filename to save the image to disk")
        else:
            fig.savefig(img_folder + img_filename, dpi=300, bbox_inches='tight')

    plt.show()
    
    if save_img == True and img_filename is not None:
        print(f"Saved {img_filename} to disk")


def display_cov_matrix(cov_matrix, centered_cbar=False, title=None, save_img=False, img_filename=None, img_folder=None):
    fig, ax = plt.subplots()

    if centered_cbar == True:
        abs_val_max = max(abs(cov_matrix.min()), abs(cov_matrix.max()))
        min_val = -abs_val_max
        max_val = abs_val_max
    else: 
        min_val = cov_matrix.min()
        max_val = cov_matrix.max()
    
    # Create a heatmap with the extended colormap
    heatmap = ax.imshow(cov_matrix, cmap='RdBu_r', aspect='auto', vmin=min_val, vmax=max_val)
    
    # Add a colorbar
    cbar = fig.colorbar(heatmap)
   

    if title is not None:
        plt.title(title, fontsize=10)
    else:
        plt.title(f"Covariance Matrix\n(Pairwise Node Covariance at Lag 0)", fontsize=10)

    if save_img == True:
        if img_filename is None:
            print("Please provide a filename to save the image to disk")
        else:
            fig.savefig(img_folder + img_filename, dpi=300, bbox_inches='tight')

    plt.show()
    
    if save_img == True and img_filename is not None:
        print(f"Saved {img_filename} to disk")



def draw_graph_basic(G, title, save_img=False, img_filename=None, img_folder=None):
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')

    # plt.figure(figsize=(16, 8))
    plt.figure(figsize=(8, 4))

    # Draw the nodes of the graph G
    nx.draw_networkx_nodes(G, pos, node_color="cornflowerblue", node_size=10, node_shape='o')

    # label a couple nodes only
    # nodes_to_label = [0, G.number_of_nodes()-1]
    # labels = {node: labels[node] for node in nodes_to_label}
    # nx.draw_networkx_labels(G, pos, labels, font_weight='normal', font_color='black', font_size=10)

    # Draw all edges with default style
    # nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='grey', alpha=0.5, arrows=True,  connectionstyle='arc')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', width=0.4, alpha=1)

    # Remove axis lines and ticks
    # plt.axis('off')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    min_lon, max_lon, min_lat, max_lat = get_min_lon_lat(G)
    plt.xlabel(f"Longitudes \n({min_lon}° to {max_lon}° E)")
    plt.ylabel(f"Latitutdes \n({min_lat}° to {max_lat}°)")

    # add a margin around the plot
    plt.gca().margins(0.0)

    # adjusts the spacing around the plot to ensure labels are not cut off.
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    if save_img == True:
        if img_filename is None:
            print("Please provide a filename to save the image to disk")
        else:
            plt.savefig(img_folder + img_filename.split(".jpg")[0] + "_no_title.jpg", dpi=300, bbox_inches='tight')


    plt.title(title, fontsize=10)
    
    if save_img == True:
        if img_filename is None:
            print("Please provide a filename to save the image to disk")
        else:
            plt.savefig(img_folder + img_filename, dpi=300, bbox_inches='tight')

    plt.show()
    
    if save_img == True and img_filename is not None:
        print(f"Saved {img_filename} to disk")


def draw_graph_with_feat_vals(G, feat_name, title=None, cbar_label=None, draw_edges=False, save_img=False, img_filename=False, img_folder=None):
    if feat_name == "sst_anom":
        feat_num = 0
    # Extract node attributes
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')
    feat_val_dict = nx.get_node_attributes(G, 'x')

    # for key, val in feat_val_dict.items():
    #     print(key, val)

    feat_vals = [feat[feat_num] for feat in feat_val_dict.values()]

    # Normalize the sst values for the heatmap
    norm = mcolors.Normalize(vmin=min(feat_vals), vmax=max(feat_vals))
    cmap = cm.coolwarm

    # Map sst values to colors
    node_colors = [cmap(norm(feat_val_dict[node][feat_num])) for node in G.nodes()]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(14, 6))
    # plt.figure(figsize=(8, 6))

    ax.axis('off')

    # Draw the graph with node colors based on sst values
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30)

    if draw_edges == True:
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='grey', alpha=1)

    # Draw the node labels
    # nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_weight='bold')

    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)

    if cbar_label is not None:
        cbar.set_label(f'{cbar_label}')
    else:
        cbar.set_label(f'{feat_name} value')

    if title is not None:
        plt.title(title)
    else:
        plt.title(f"Graph of {feat_name}\n")

    if save_img == True:
        fig.savefig(img_folder + img_filename, dpi=300, bbox_inches='tight')

    plt.show()
    
    if save_img == True:
        print(f"Saved {img_filename} to disk")


def draw_isolated_nodes(G, title, hide_isolated=False, save_img=False, img_filename=None, img_folder=None):
    pos = nx.get_node_attributes(G, 'pos')

    all_nodes = list(G.nodes)
    isolated_nodes = get_isolated_nodes(G)
    connected_nodes = [node for node in all_nodes if node not in isolated_nodes]

    plt.figure(figsize=(12, 4))

    # draw connected nodes and isolated nodes in different colors
    nx.draw_networkx_nodes(G, pos, nodelist=connected_nodes, node_size=10, node_color="cornflowerblue")

    if hide_isolated:
        nx.draw_networkx_nodes(G, pos, nodelist=isolated_nodes, node_size=10, node_color="white")
    else:
        nx.draw_networkx_nodes(G, pos, nodelist=isolated_nodes, node_size=10, node_color="black")

    # Remove axis lines and ticks
    # plt.axis('off')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    min_lon, max_lon, min_lat, max_lat = get_min_lon_lat(G)
    plt.xlabel(f"Longitudes \n({min_lon}° to {max_lon}° E)")
    plt.ylabel(f"Latitutdes \n({min_lat}° to {max_lat}°)")

    # add a margin around the plot
    plt.gca().margins(0.02)

    plt.title(title)
    # adjusts the spacing around the plot to ensure labels are not cut off.
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    if save_img == True:
        if img_filename is None:
            print("Please provide a filename to save the image to disk")
        else:
            plt.savefig(img_folder + img_filename, dpi=300, bbox_inches='tight')

    plt.show()
    
    if save_img == True and img_filename is not None:
        print(f"Saved {img_filename} to disk")


def draw_bfs_graph(G, title, start_node=False):
    # Position nodes according to breadth-first search algorithm.
    if start_node == False:
        #Compute the degree of each node
        degrees = dict(G.degree())
        
        # Find the node with the highest degree
        max_degree_node = max(degrees, key=degrees.get)
        start_node = max_degree_node

    min_degree_node = min(degrees, key=degrees.get)
    
    plt.figure(figsize=(18, 12))
    pos = nx.bfs_layout(G, start=start_node, align='vertical', scale=1, center=None)  # positions for all nodes
    
    # # Adjust positions to spread nodes more
    # scale_factor = 2  # Adjust this factor to spread nodes more or less
    # for node in pos:
    #     pos[node] = (pos[node][0] * scale_factor, pos[node][1] * scale_factor)

    
    
    # Draw the nodes of the graph G
    nx.draw_networkx_nodes(G, pos, node_color="cornflowerblue", node_size=10, node_shape='o')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', width=0.4, alpha=0.3)

    # label a couple nodes only
    labels = nx.get_node_attributes(G, 'label')
    nodes_to_label = [start_node, min_degree_node]
    labels = {node: labels[node] for node in nodes_to_label}
    nx.draw_networkx_labels(G, pos, labels, font_weight='normal', font_color='black', font_size=8)
    
    
    plt.title(title)
    plt.show()
    

def filter_nodes_by_min_degree(G, min_degree):
    low_degree_nodes = get_low_degree_nodes(G, min_degree)
    print(f"Removed {len(low_degree_nodes)} nodes with degree < {min_degree} from {G.number_of_nodes()} original nodes")
    G_min_degree = G.copy()
    G_min_degree.remove_nodes_from(low_degree_nodes)
    return G_min_degree


def find_file(directory, search_str, file_extension=None):
    filenames = os.listdir(directory)
    
    # Iterate over files in the directory and return the first match
    for filename in filenames:
        if search_str in filename:
            # Check if the file has the specified extension (if provided)
            if file_extension:
                if filename.endswith(file_extension):
                    filepath = os.path.join(directory, filename)
                    return filename, filepath
            else:
                # If no extension filter is provided, return the first match
                filepath = os.path.join(directory, filename)
                return filename, filepath
                
    print(f"No file with search string '{search_str}' and extension '{file_extension}' was found")
    return None


def get_largest_connected_component(G):
    return max(nx.connected_components(G), key=len)
    
    
def get_component_with_most_edges(G):
    return max(nx.connected_components(G), key=lambda component: G.subgraph(component).number_of_edges())


def get_isolated_nodes(G):
    return list(nx.isolates(G))


def get_low_degree_nodes(G, min_degree):
    low_degree_nodes = [node for node in G.nodes() if G.degree(node) < min_degree]
    return low_degree_nodes


def get_min_lon_lat(G):
    pos = nx.get_node_attributes(G, 'pos')
    pos_tuples = list(pos.values())
    reverse_pos_dict = get_reverse_pos_dict(G)

    # Compute the min and max lon and lat
    min_lon = min(t[0] for t in pos_tuples)
    max_lon = max(t[0] for t in pos_tuples)

    min_lat = min(t[1] for t in pos_tuples)
    max_lat = max(t[1] for t in pos_tuples)

    return min_lon, max_lon, min_lat, max_lat


def get_nan_points(da: xr.DataArray):
    """
    Return indices and coordinates of points that have NaN values accross all times for its original time step (last entry of the window: win[2])
    da : DataArray with coordinates ['time', 'window', 'lon', 'lat']
    """

    # Stack the lon and lat dimensions
    stacked_da = da.stack(lonlat=('lon', 'lat'))

    # Identify points that are NaN for every time step
    nan_mask_all_times = stacked_da[:,-1,:].isnull().any(dim='time')

    # Identify NaN values and their indices in the stacked DataArray
    nan_indices = np.where(nan_mask_all_times)[0]

    # Extract the corresponding coordinates
    nan_coords = stacked_da.lonlat[nan_indices].values

    return nan_indices, nan_coords


def get_zero_variance_points(da: xr.DataArray):
    """
    Return indices and coordinates of points that have the same value accross all times
    da : DataArray with coordinates ['time', 'lon', 'lat']
    """

    # Stack the lon and lat dimensions
    stacked_da = da.stack(lonlat=('lon', 'lat'))

    # Convert to a numpy array
    stacked_data_np = stacked_da[:,-1,:].values
    
    # Calculate the variance along the time dimension
    variance = np.var(stacked_data_np, axis=0)

    # Identify points that have zero variance
    zero_var_indices = np.where(variance == 0)[0]

    # Extract the corresponding coordinates
    zero_var_coords = stacked_da.lonlat[zero_var_indices].values
    
    return zero_var_indices, zero_var_coords


def get_nan_points_per_date(da: xr.DataArray, date_selection):
    """
    da : DataArray with coordinates ['time', 'lon', 'lat']
    """
    da = da.sel(time=date_selection)

    # Stack the lon and lat dimensions
    stacked_da = da.stack(lonlat=('lon', 'lat'))

    # Identify NaN values and their indices in the stacked DataArray
    nan_mask = np.isnan(stacked_da)
    nan_indices = np.where(nan_mask)[0]

    # Extract the corresponding coordinates
    nan_coords = stacked_da.lonlat[nan_indices].values

    return nan_indices, nan_coords


def get_neighbor_locs(G, target_lon, target_lat):
    pos = nx.get_node_attributes(G, 'pos')
    reversed_pos_dict = {value: key for key, value in pos.items()}

    if (target_lon, target_lat) not in reversed_pos_dict:
        print(f"Target ({target_lon, target_lat}) doesn't exit in Graph. Return empty neighbors list.")
        return [], [], []
        
    target_idx = reversed_pos_dict[(target_lon, target_lat)]
    neighbors_idx = sorted(set(G.neighbors(target_idx)))

    # print(f"target_idx: {target_idx}, coordinates (lon, lat): ({target_lon}, {target_lat})")
    # print(f"neighbors_idx: \n{neighbors_idx}")

    neighbor_lons = [pos[neighbor_idx][0] for neighbor_idx in neighbors_idx]
    neighbor_lats = [pos[neighbor_idx][1] for neighbor_idx in neighbors_idx]

    return neighbor_lons, neighbor_lats, neighbors_idx


def get_node_features_ts(data_array: xr.DataArray, fillna=False) -> torch.Tensor:
    """
    Converts an xarray.DataArray to a time series node feature tensor.
    
    Parameters:
    - data_array (xr.DataArray): The input data array containing node features for each combination of lat / lon dims
    
    Returns:
    - torch.Tensor: The resulting time series node features tensor with shape (num_time, num_nodes, num_features)
    """

    # print("data_array.shape:           ", data_array.shape)

    if fillna:
        print("Replace NaNs with 0s")
        data_array = data_array.fillna(0)  # Replace NaNs with 0

    # Stack lat and lon dimensions into a single lonlat new dimension.
    # (time, lat, lon) --> (time, lonlat)
    flattened_data_array = data_array.stack(points=('lon', 'lat'))
    # print("flattened_data_array.shape: ", flattened_data_array.shape)

    # Convert to a tensor
    flattened_data_tensor = torch.from_numpy(flattened_data_array.values)
    # print("flattened_data_tensor.shape: ", flattened_data_tensor.shape)

    # Add an extra dimension to represent num_features
    # node_features_ts = flattened_data_tensor.unsqueeze(-1)
    # print("node_features_ts.shape after unsqueeze: ", node_features_ts.shape)

    return flattened_data_tensor


def get_nodes_with_nan(G):
    nodes_with_nan_values = []
    for node in G.nodes:
        if np.isnan(G.nodes[node]['x']):
            nodes_with_nan_values.append(node)

    return nodes_with_nan_values


def get_pos_dict(G):
    pos = nx.get_node_attributes(G, 'pos')
    return pos


def get_reverse_pos_dict(G):
    """
    Helps to find a node's id from it's positional coordinates, e.g.:

    pos[500] --> (140, -54)
    reverse_pos_dict[((140, -54))] --> 500
    """
    # Get graph's positions attribute (a dictionary with nodes as keys and positions as values)
    # e.g. [(0, (0, -3)), (1, (1, -3)), (2, (2, -3)), (3, (3, -3)), ...]
    pos = nx.get_node_attributes(G, 'pos')

    # e.g. [((0, -3), 0), ((1, -3), 1), ((2, -3), 2), ((3, -3), 3), ...]
    reverse_pos_dict = {value: key for key, value in pos.items()}
    return reverse_pos_dict


def get_time_index(xarray, date):
    date_array = xarray.time.values.astype('datetime64[D]')
    date_list = list(date_array)
    specified_date = np.datetime64(date, 'D')
    return date_list.index(specified_date)


def infer_V0(model, test_dataloader, verbose=True):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if verbose:
        print("Device used for testing: ", device)
        print("Size of dataloader: ", len(test_dataloader.dataset))
        print("Number of batches in dataloader: ", len(test_dataloader))

    model.eval()
    model.to(device)

    # calculates the predictions of the best saved model
    predictions = np.asarray([])
    for i, batch in enumerate(test_dataloader):
        batch = batch.to(device)
        batch_predictions = model(batch.x, batch.edge_index, batch.batch).squeeze()
        if len(batch_predictions.size()) == 0:
            batch_predictions = torch.Tensor([batch_predictions])

        # Move the batch_predictions tensor to CPU, convert it to NumPy array, and concatenate it to the rest of the predictions
        predictions = np.concatenate([predictions, batch_predictions.detach().cpu().numpy()])
    return predictions


def infer(model, test_dataloader, verbose=True):
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
            batch = batch.to(device)
            batch_predictions = model(batch.x, batch.edge_index, batch.batch).squeeze()
            if len(batch_predictions.size()) == 0:
                batch_predictions = batch_predictions.unsqueeze(0)
            
            # Append batch predictions to the list
            predictions.append(batch_predictions.detach().cpu().numpy())
            
    # Concatenate all batch predictions into a single array
    predictions = np.concatenate(predictions)
    return predictions



def load_model(model, optimizer, experiment_name, target, device, verbose=False):
    if "CNN" in experiment_name:
        model_type = "cnn"
    elif "GCN" in experiment_name:
        model_type = "gcn"
    else:
        print(f"Unknown model type in experiment_name {experiment_name}. Abort")
        return
    
    model_subfolder = experiment_name.split("_Lead")[0]


    if target == "oni":
        print(f"• Loading ONI model {experiment_name}.pt")
        load_path = f'{MODELS_ONI_FOLDER}/{model_type}/{model_subfolder}/{experiment_name}.pt'
        load_model_state(model, optimizer, load_path, device)

    elif target == "nino34":
        # print(f"• Loading Nino34 model {experiment_name}.pt")
        load_path = f'{MODELS_NINO34_FOLDER}/{model_type}/{model_subfolder}/{experiment_name}.pt'
        load_model_state(model, optimizer, load_path, device)

    elif target == "E":
        # print(f"• Loading E model {experiment_name}.pt")
        load_path = f'{MODELS_EC_FOLDER}/{model_type}/{model_subfolder}/{experiment_name}.pt'
        load_model_state(model, optimizer, load_path, device)

    elif target == "C":
        # print(f"• Loading C model {experiment_name}.pt")
        load_path = f'{MODELS_EC_FOLDER}/{model_type}/{model_subfolder}/{experiment_name}.pt'
        load_model_state(model, optimizer, load_path, device)
    else:
        print("• No matching target found. No model loaded")
        return
    
    if verbose==True:
        print(f"Load model for target {target}:")
        print(f"Experiment name: {experiment_name}.pt")


def load_model_state(model, optimizer, load_path, device):
    # print(f"load_model_state to device {device}")
    state = torch.load(load_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    
    # Ensure optimizer's state tensors are on the correct device
    for state_idx, state in enumerate(optimizer.state.values()):
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                # Check if the tensor is on the correct device
                if v.device != device:
                    # print(f"Tensor in state {state_idx} with key '{k}' is on {v.device}, expected {device}")
                    # Move the tensor to the correct device
                    state[k] = v.to(device)
    # return state, model, optimizer


def plot_coords(coordinates_to_plot, figsize=(4, 4), ms=10, marker='o', color='red', norm=None, ax=None, projection="Orthographic", title=None,
                save_img=None, img_filename=None, img_folder=None):
    """
    coordinates_to_plot: list of tuples of coordinates (lat, lon)
    """
    # Extract the lat and lon values from the coordinates
    lon_list = [coord[0] for coord in coordinates_to_plot]
    lat_list = [coord[1] for coord in coordinates_to_plot]

    # Create a map plot
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(projection=ccrs.Miller())

    # Add features to the map
    ax.coastlines()
    ax.set_global()
    ax.stock_img()

    transform = ccrs.PlateCarree()
    ax.scatter(lon_list, lat_list, marker=marker, s=ms, color=color, transform=transform)

    if title is None:
        ax.set_title('Nodes of Coordinates', fontsize=10)
    else:
        ax.set_title(title, fontsize=10)
    plt.show()


def plot_data_no_graph(data, suptitle=None, title=None, cbar_label=None, save_img=True, img_folder=False, img_filename=False):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 4))

    im = data.plot(ax=ax, add_colorbar=False, cmap='RdBu_r')

    # Set title to the dataset's name
    if title is not None:
        ax.set_title(title, fontsize=10)

    # Accessing axis labels
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    ax.set(xlabel=None, ylabel=None)

    # Set title
    if suptitle is not None:
        fig.suptitle(suptitle, fontweight="bold")

    # Create a shared colorbar for the entire figure
    fig.subplots_adjust(right=0.8, hspace=0.4)  # Adjust spacing between subplots, as well as right margin for the colorbar
    cbar_ax = fig.add_axes([0.82, 0.10, 0.02, 0.78])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label)

    fig.text(0.5, 0.0, xlabel, ha='center', va='center', fontsize=9)
    fig.text(0.07, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=9)

    if save_img == True:
        fig.savefig(img_folder + img_filename, dpi=300, bbox_inches='tight')

    plt.show()
    
    if save_img == True:
        print(f"Saved {img_filename} to disk")


def plot_degree_histogram(G, yscale='linear', title=None, save_img=False, img_filename=None, img_folder=None):
    # code adapted from networkx documentation
    # https://networkx.org/documentation/stable/auto_examples/drawing/plot_degree.html
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    unique_degrees, counts = np.unique(degree_sequence, return_counts=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(unique_degrees, counts)
    if title is not None:
        ax.set_title(title, fontsize=10)
    else:
        ax.set_title("Degree histogram", fontsize=10)
    ax.set_xlabel("Node Degree")
    if yscale == "log":
        ax.set_ylabel("Log of # of Nodes")
    else:
        ax.set_ylabel("# of Nodes")
    ax.set_yscale(yscale)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)

    if save_img == True:
        if img_filename is None:
            print("Please provide a filename to save the image to disk")
        else:
            plt.savefig(img_folder + img_filename, dpi=300, bbox_inches='tight')
            print(f"Saved {img_filename} to disk")
    plt.show()


def plot_neighbors(G, target_lon, target_lat, figsize=(4, 4), ms=50, title=None, neighbor_clr='r', target_clr='k', edge_clr='w', marker='o',
                   cmap=None, norm=None, ax=None):
    neighbor_lons, neighbor_lats, _ = get_neighbor_locs(G, target_lon, target_lat)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(projection=ccrs.Orthographic(central_longitude=target_lon, central_latitude=target_lat))

    ax.set_global()
    ax.stock_img()
    if title is None:
        ax.set_title(f'Neighbors of node ({target_lon}, {target_lat})', fontsize=10)
    else:
        ax.set_title(title, fontsize=10)

    transform = ccrs.PlateCarree()
    if cmap is not None:
        ax.scatter(neighbor_lons, neighbor_lats, marker=marker, s=ms, c=neighbor_clr, edgecolor=edge_clr, transform=transform, cmap=cmap, norm=norm)
    else:
        ax.scatter(neighbor_lons, neighbor_lats, marker=marker, s=ms, c=neighbor_clr, edgecolor=edge_clr, transform=transform)
    ax.scatter(target_lon, target_lat, marker=marker, s=ms, c=target_clr, edgecolor=edge_clr, transform=transform)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='white', alpha=0.2, linestyle='-')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

    return ax


def plot_neighbors_corr(G, target_lon, target_lat, reversed_pos_dict, correlation_matrix, time_idx_range=None, figsize=(4, 4), ms=50, title=None,
                        cmap='RdBu_r', target_clr='w', edge_clr='w', marker='o', levels=np.linspace(-1, 1, 21), ax=None, save_img=False,
                        img_folder=None):
    _, _, neighbor_idx = get_neighbor_locs(G, target_lon, target_lat)
    target_idx = reversed_pos_dict[(target_lon, target_lat)]

    corrs = correlation_matrix[target_idx, neighbor_idx]

    cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    ax = plot_neighbors(G, target_lon, target_lat, figsize=figsize, ms=ms, title=title, neighbor_clr=corrs, target_clr=target_clr, edge_clr=edge_clr,
                        marker=marker, cmap=cmap, norm=norm, ax=ax)

    if save_img == True:
        target_str = f"{str(target_lon).replace('.', '-')}-{str(target_lat).replace('.', '-')}"
        filename = f"neighbors_of_{target_str}.jpg"
        plt.savefig(img_folder + filename, dpi=300, bbox_inches='tight')
        print(f"Saved {filename} to disk")

    return ax


# chat gpt examples for save and load
def plot_neighbors_corr_4x4(G, target_lon_list, target_lat_list, region_names, reverse_pos_dict, sst_corr_matrix, subtitle_short, img_filename_tag, source_id, lag,
                            projection="Orthographic", central_lon=None, central_lat=None, save_img=False, img_folder=None):
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))

    # Flatten the axes array for easy indexing in the loop
    axs = axs.flatten()

    # Plot each graph in the corresponding subplot
    for i in range(4):
        axs[i].axis('off')
        title = (f'{region_names[i]} region\n'
                 f'Neighbors of node ({target_lon_list[i]}, {target_lat_list[i]})\n'
                 f'({subtitle_short} at Lag {lag}, 5x5 Res)')
        # title = f'Neighbors of node ({target_lon_list[i]}, {target_lat_list[i]})\n(One-Hop Neighbors, 5x5 Resolution)'

        if projection == "Orthographic":
            if central_lon is not None and central_lat is not None:
                # print the same central longitude and latitude for all plots
                axs[i] = plt.subplot(2, 2, i + 1, projection=ccrs.Orthographic(central_longitude=central_lon, central_latitude=central_lat))
            else:
                axs[i] = plt.subplot(2, 2, i + 1,
                                     projection=ccrs.Orthographic(central_longitude=target_lon_list[i], central_latitude=target_lat_list[i]))
        elif projection == "Mollweide":
            if central_lon is not None:
                axs[i] = plt.subplot(2, 2, i + 1, projection=ccrs.Mollweide(central_longitude=central_lon))
            else:
                axs[i] = plt.subplot(2, 2, i + 1, projection=ccrs.Mollweide())
        elif projection == "Robinson":
            if central_lon is not None:
                axs[i] = plt.subplot(2, 2, i + 1, projection=ccrs.Robinson(central_longitude=central_lon))
            else:
                axs[i] = plt.subplot(2, 2, i + 1, projection=ccrs.Robinson(central_longitude=target_lon_list[i]))
        elif projection == "AlbersEqualArea":
            if central_lon is not None and central_lat is not None:
                axs[i] = plt.subplot(2, 2, i + 1, projection=ccrs.AlbersEqualArea(central_longitude=central_lon, central_latitude=central_lat))
            else:
                axs[i] = plt.subplot(2, 2, i + 1, projection=ccrs.AlbersEqualArea(central_longitude=target_lon_list[i]))
        elif projection == "Miller":
            if central_lon is not None:
                axs[i] = plt.subplot(2, 2, i + 1, projection=ccrs.Miller(central_longitude=central_lon))
            else:
                axs[i] = plt.subplot(2, 2, i + 1, projection=ccrs.Miller(central_longitude=target_lon_list[i]))
        else:
            print(f"The projection argument \'{projection}\' is invalid")

        plot_neighbors_corr(G, target_lon_list[i], target_lat_list[i], reverse_pos_dict, sst_corr_matrix, title=title, ax=axs[i],
                            img_folder=img_folder)

    # Create a colormap and normalization for the colorbar
    levels = np.linspace(-1, 1, 21)
    cmap = plt.get_cmap('RdBu_r')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    levels = np.linspace(-1, 1, 21)
    cbar_ax = fig.add_axes([0.36, 0.08, 0.3, 0.03])  # [left, bottom, width, height]
    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(smap, cax=cbar_ax, orientation='horizontal')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.set_title('Correlation')

    # Adjust layout
    fig.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.9, hspace=0.4, wspace=0.1)

    if save_img == True:
        filename = f"neighbors_{img_filename_tag}_{projection}_{source_id}.jpg"
        fig.savefig(img_folder + filename, dpi=300, bbox_inches='tight')

    # Display the combined plot
    plt.show()
    
    if save_img == True:
        print(f"Saved {filename} to disk")


def plot_nino_time_series(y, predictions, source_id, title):
    """
    inputs
    ------
        y           pd.Series : time series of the true Nino index
        predictions np.array  : time series of the predicted Nino index (same
                                length and time as y)
        titile                : the title of the plot

    outputs
    -------
        None.  Displays the plot
    """
    predictions = pd.Series(predictions, index=y.index)
    predictions = predictions.sort_index()
    y = y.sort_index()

    plt.figure(figsize=(6, 1))  # Set the size of the figure

    plt.plot(y, label=f'GT ({source_id})')
    plt.plot(predictions, '--', label='ML Predictions')
    # plt.legend(loc='best', fontsize=7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)

    plt.title(title, fontsize=8)
    plt.ylabel('ONI')
    # plt.xlabel('Date')
    plt.show()
    plt.close()


def plot_node_neighbors(G, lon_point, lat_point, title=None):
    """
    Plot neighbors of specified location
    """
    pos = nx.get_node_attributes(G, 'pos')
    reverse_pos_dict = get_reverse_pos_dict(G)
    node_of_interest = reverse_pos_dict[(lon_point, lat_point)]
    neighbors = set(G.neighbors(node_of_interest))
    # print(neighbors)
    # [print(neighbor, pos[neighbor]) for neighbor in neighbors]
    # print()

    edges_to_neighbors = [(node_of_interest, neighbor) for neighbor in neighbors]
    # print("edges_to_neighbors: ", edges_to_neighbors)

    neighbors.add(node_of_interest)  # include the node of interest in the subgraph
    neighbors_subgraph = G.subgraph(neighbors)
    # print(neighbors_subgraph)

    # Visualize the subgraph
    plt.figure(figsize=(16, 9))
    subG_labels = nx.get_node_attributes(neighbors_subgraph, 'label')
    subG_pos = nx.get_node_attributes(neighbors_subgraph, 'pos')

    # Create a color map
    node_colors = ['red' if node == node_of_interest else 'skyblue' for node in neighbors_subgraph.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(neighbors_subgraph, subG_pos, node_color=node_colors, node_size=500)

    # Draw labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(neighbors_subgraph, subG_pos, subG_labels, font_color='black', font_size=5)

    # Draw all edges with default style
    # nx.draw_networkx_edges(neighbors_subgraph, subG_pos, edgelist=neighbors_subgraph.edges(), edge_color='grey', alpha=0.5)

    # # Draw edges in the shortest paths with a bolder line
    nx.draw_networkx_edges(neighbors_subgraph, subG_pos, edgelist=edges_to_neighbors, edge_color='black', width=1)

    if title:
        plt.title(title)
    else:
        plt.title(f"Neighbors of node {pos[node_of_interest]}")

    plt.show()


def plot_nodes(G, central_lon=None, central_lat=None, figsize=(4, 4), ms=20, neighbor_clr='k', target_clr='k', edge_clr='w', marker='o',
               cmap=plt.cm.viridis, norm=None, ax=None, projection="Orthographic", title=None, save_img=None, img_filename=None, img_folder=None):
    full_pos_dict = get_pos_dict(G)
    node_ids = G.nodes
    tuples_list = [full_pos_dict[node_id] for node_id in node_ids]
    lon_list, lat_list = zip(*tuples_list)

    lon_list = list(lon_list)
    lat_list = list(lat_list)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        if projection == "Orthographic":
            if central_lon is not None and central_lat is not None:
                # print the same central longitude and latitude for all plots
                ax = plt.subplot(projection=ccrs.Orthographic(central_longitude=central_lon, central_latitude=central_lat))
            else:
                ax = plt.subplot(projection=ccrs.Orthographic())
        elif projection == "Mollweide":
            if central_lon is not None:
                ax = plt.subplot(projection=ccrs.Mollweide(central_longitude=central_lon))
            else:
                ax = plt.subplot(projection=ccrs.Mollweide())
        elif projection == "Robinson":
            if central_lon is not None:
                ax = plt.subplot(projection=ccrs.Robinson(central_longitude=central_lon))
            else:
                ax = plt.subplot(projection=ccrs.Robinson())
        elif projection == "AlbersEqualArea":
            if central_lon is not None and central_lat is not None:
                ax = plt.subplot(projection=ccrs.AlbersEqualArea(central_longitude=central_lon, central_latitude=central_lat))
            else:
                ax = plt.subplot(projection=ccrs.AlbersEqualArea())
        elif projection == "Miller":
            if central_lon is not None:
                ax = plt.subplot(projection=ccrs.Miller(central_longitude=central_lon))
            else:
                ax = plt.subplot(projection=ccrs.Miller())
        else:
            print(f"The projection argument \'{projection}\' is invalid")

    # Create a color map based on degree
    degrees = dict(G.degree())
    max_degree = max(degrees.values())

    norm = plt.Normalize(vmin=0, vmax=max_degree)
    colors = [cmap(norm(degree)) for degree in degrees.values()]

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.030, pad=0.04)
    cbar.set_label('Node Degree')

    # cbar.set_ticks(range(int(max_degree) + 1))

    ax.set_global()
    ax.stock_img()



    transform = ccrs.PlateCarree()
    if cmap is not None:
        ax.scatter(lon_list, lat_list, marker=marker, s=ms, c=colors, edgecolor=edge_clr, transform=transform)
    else:
        ax.scatter(lon_list, lat_list, marker=marker, s=ms, c=colors, edgecolor=edge_clr, transform=transform)

    # Adjust layout
    # fig.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.9, hspace=0.4, wspace=0.1)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='white', alpha=0.2, linestyle='-')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

    
    if save_img == True:
        # Save plot without the title
        fig.savefig(img_folder + img_filename.split(".jpg")[0] + "_no_title.jpg", dpi=300, bbox_inches='tight')

    if title is None:
        ax.set_title(f'Nodes', fontsize=10)
    else:
        ax.set_title(title, fontsize=10)

    if save_img == True:
        # Save plot with the title
        fig.savefig(img_folder + img_filename, dpi=300, bbox_inches='tight')

    plt.show()
    
    if save_img == True:
        print(f"Saved {img_filename} to disk")



    
    

def print_isolated_vs_connected_stats(G):
    all_nodes = list(G.nodes)
    isolated_nodes = get_isolated_nodes(G)
    connected_nodes = [node for node in all_nodes if node not in isolated_nodes]

    print(f"Number of isolated nodes:  {len(isolated_nodes):>4} / {G.number_of_nodes()} ({len(isolated_nodes) / G.number_of_nodes():.2%})")
    print(f"Number of connected nodes: {len(connected_nodes):>4} / {G.number_of_nodes()} ({len(connected_nodes) / G.number_of_nodes():.2%})")


def remove_isolated_nodes(G):
    isolated_nodes = get_isolated_nodes(G)
    print(f"Removed {len(isolated_nodes)} isolated nodes")
    G_connected = G.copy()

    G_connected.remove_nodes_from(isolated_nodes)
    return G_connected

def remove_nodes_from(G, nodes_to_remove):
    G_removed = G.copy()
    G_removed.remove_nodes_from(nodes_to_remove)
    print(f"Removed {len(nodes_to_remove)} nodes from Graph")
    return G_removed


def remove_nan_nodes(G):
    nan_nodes = get_nodes_with_nan(G)
    print(f"Removed {len(nan_nodes)} NaN nodes")
    G_no_NaN = G.copy()
    G_no_NaN.remove_nodes_from(nan_nodes)
    return G_no_NaN


def save_model_state(model, optimizer, save_path):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, save_path)
    # print(f"Saved model to {save_path}")


def analyze_gradients(model, device, dataloader, criterion, verbose=False):
    """
    Key changes to train_epoch function (the focus is on computing gradients)
    1. Put Model in Evaluation Mode
    2. No Optimization Step
    3. Calculate Gradient (set requires_grad=True on input features, and call batch_loss.backward()
        => the gradients of the loss with respect to the input features are calculated.
    """
    model = model.to(device)
    model.eval()  # Ensure the model is in evaluation mode 

    all_gradients = []
    total_loss = 0.0
    total_samples = 0

    all_gradients = []

    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        batch_size = batch.x.size(0)
        total_samples += batch_size
        
        if verbose==True:
            print(f"batch {i} number of graphs: {batch.num_graphs}")    
            print(f"batch {i} x.shape: {batch.x.shape} (node features for all the graphs in the current batch, concatenated into a single tensor)")
            print(f"batch {i} x.size(0): {batch.x.size(0)} (total number of nodes across all graphs in the batch.)")
            

        # Enable gradients for the input features 
        batch.x.requires_grad = True 

        # Forward pass 
        
        batch_predictions = model(batch.x, batch.edge_index, batch.batch).squeeze()
        batch_loss = criterion(batch_predictions, batch.y.squeeze())

        # Backward pass to compute gradients
        batch_loss.backward() 
        
        # Collect the gradients for analysis 
        if verbose == True:
            print("batch.x.grad.shape :", batch.x.grad.shape)
            print()

        all_gradients.append(batch.x.grad.detach().cpu().numpy()) 

        # Add this batch loss to the running loss and append to history
        total_loss += batch_loss.item() * batch_size  # Weight by batch size

        # Clear the gradients for the next batch 
        batch.x.grad.zero_() 
        
    if verbose==True:
        print(f"total_samples: {total_samples:>7,} (total number of nodes across all graphs in the epoch.)")
        
    # Weighted average loss per sample
    avg_loss = total_loss / total_samples  

    # Aggregate gradients for all batches 
    concatenated_nodes_gradients = np.concatenate(all_gradients, axis=0) 

    return avg_loss, concatenated_nodes_gradients





def train_epoch(model, device, train_dataloader, optimizer, epoch, criterion):
    model.train()        
    train_epoch_running_loss = 0.0
    train_epoch_loss_history = []
    train_total_samples = 0
    
    for i, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        batch_size = batch.x.size(0)
        train_total_samples += batch_size
        
        optimizer.zero_grad() # zero the parameter gradients

        # Forward pass
        batch_predictions = model(batch.x, batch.edge_index, batch.batch).squeeze()
        batch_loss = criterion(batch_predictions, batch.y.squeeze())
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
def validate_epoch(model, device, val_dataloader, criterion):
    model.eval()
    val_epoch_running_loss = 0.0
    val_epoch_loss_history = []
    val_total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            batch = batch.to(device)
            batch_size = batch.x.size(0)
            val_total_samples += batch_size

            # Forward pass
            batch_predictions = model(batch.x, batch.edge_index, batch.batch).squeeze()
            batch_loss = criterion(batch_predictions, batch.y.squeeze())
            # print(f"Val batch_loss ({batch_size} samples): {batch_loss}")

            
            # Add this batch loss to the running loss and append to history
            val_epoch_running_loss += batch_loss.item() * batch_size  # Weight by batch size
            val_epoch_loss_history.append(batch_loss.item())
            
    # val_epoch_avg_loss = val_epoch_running_loss / len(val_dataloader) # average loss per batch
    val_epoch_avg_loss = val_epoch_running_loss  / val_total_samples  # Weighted average loss per sample
    # print(f"-----Val epoch_avg_loss: {val_epoch_avg_loss}")
    return val_epoch_avg_loss, val_epoch_loss_history


def train_network(model, criterion, optimizer, train_dataloader, val_dataloader,
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

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if verbose:
        print("Prediction target: ", target)
        print("Device used for training: ", device)
    model = model.to(device)

    if "CNN" in experiment_name:
        model_type = "cnn"
    elif "GCN" in experiment_name:
        model_type = "gcn"
    else:
        print(f"Unknown model type in experiment_name {experiment_name}. Abort")
        return


    # create 1 folder for all models of the same architecture with different leads
    model_subfolder = experiment_name.split("_Lead")[0]
    os.makedirs(f'{model_folder}/{model_type}/{model_subfolder}', exist_ok=True) 
            
    if save_model:
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
        train_epoch_avg_loss, train_epoch_loss_history = train_epoch(model, device, train_dataloader, optimizer, epoch, criterion)
        train_losses.append(train_epoch_avg_loss)
        train_loss_history.extend(train_epoch_loss_history)

        ##### Validation phase #####
        val_epoch_avg_loss, val_epoch_loss_history = validate_epoch(model, device, val_dataloader, criterion)
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




def plot_distribution(dataset_list: Dict[str, xr.Dataset], var_name: str, bins, start_date, end_date, min_lat, max_lat, min_lon, max_lon, resolution, subtitle=None, save_img=False, img_filename=None, img_folder=None):
    
    # sst_anom_min, sst_anom_max = np.inf, -np.inf
    # max_frequency_per_bin = 0

    # # Calculate max value sst anoms and max frequency accross all datasets
    # for ds in dataset_list.values():
    #     data = ds[var_name].sel(time=slice(start_date, end_date),
    #                             lat=slice(min_lat, max_lat),
    #                             lon=slice(min_lon, max_lon))
        
        
    #     # Since we want the colorbar to be symetrical around zero, we need to take the weights' maximum abs value and use it for vmin and vmax
    #     sst_anom_min = min(sst_anom_min, data.min().item())
    #     sst_anom_max = max(sst_anom_max, data.max().item())
    #     print(f"min and max sst anoms: {sst_anom_min:.4f}, {sst_anom_max:.4f}")
    
    #     # Calculate frequency in the histogram
    #     counts, _ = np.histogram(data.values.flatten(), bins=bins)
    #     max_frequency_per_bin = max(max_frequency_per_bin, counts.max())
    #     print(f"max_frequency_per_bin: {max_frequency_per_bin}")

    # abs_sst_anom_max = max(abs(sst_anom_min), abs(sst_anom_max))
        
        
        
        
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(11, 4.5), sharex=True, sharey=True)

    # Iterate over datasets and their respective axes
    for (name, ds), ax in zip(dataset_list.items(), axs.flat):
        data = ds[var_name].sel(time=slice(start_date, end_date),
                                    lat=slice(min_lat, max_lat),
                                    lon=slice(min_lon, max_lon))

        data.plot.hist(ax=ax, yscale='linear', bins=bins)
        
        # Set title to the dataset's name
        ax.set_title((
            f"{name.split('.')[2]}\n"
            # f"{data.size:,} values\n"
            f"{data.count().item():,} non-nan values"
            ), 
            fontsize=10)
        ax.grid(True, linestyle='--', linewidth=0.5)
        
        # Accessing axis labels
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        ax.set(xlabel=None, ylabel=None)


    # Set title
    # Note: data.size is the total number of values for the last dataset of the input dict: assumes all datasets have the same number of values
    coordinates = utils.add_cardinals_title(min_lat, max_lat, min_lon, max_lon)
    fig.suptitle(f"SST anomalies distribution for selected datasets between {start_date}-{end_date}\n"
                f"Region {coordinates}, Resolution {resolution}x{resolution}, Total values per dataset: {data.size:,}", fontweight="bold", fontsize=10)
    

    fig.text(0.5, 0.0, 'SST Anomalies', ha='center', va='center', fontsize=9)
    fig.text(0.0, 0.5, 'Count', ha='center', va='center', rotation='vertical', fontsize=9)
    plt.tight_layout()

    if save_img == True:
        fig.savefig(img_folder + img_filename, dpi=300, bbox_inches='tight')

    plt.show()
    
    if save_img == True:
        print(f"Saved {img_filename} to disk")
        
            
    
def plot_learning_curves(model_results, target, num_epochs, train_loss_display="batch", rows_to_show=4, subtitle=None, save_img=False, img_filename=None, img_folder=None):
    
    fig, axs = plt.subplots(nrows=rows_to_show, ncols=3, figsize=(11, rows_to_show*2.5), sharex=True, sharey=True)

    # fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(12, 18), sharex=True, sharey=True)
    
    # Iterate over the linear regression models' results and their respective axes
    for (lead_time, model_dict), ax in zip(model_results[target].items(), axs.flat):

        

        # match the epoch axis scale for the train history per batch
        n_train = len(model_dict['train_loss_history']) # nb epochs * nb of train batches per train_dataloader
        t_train = num_epochs * np.arange(n_train) / n_train  
        
        # Plot Training and Validation losses
        
        if train_loss_display == "batch":
            ax.plot(t_train, model_dict["train_loss_history"], label='Train Loss (per batch)', marker='o', markersize=2, linewidth=1)
        elif train_loss_display == "epoch":
            ax.plot(range(1, len(model_dict["train_losses"])+1), model_dict["train_losses"], label='Train Loss', marker='o', markersize=3)
        else:
            print(f"Unknown train_loss_display value: {train_loss_display}. Choose 'batch' or 'epoch'")
            return
            
        ax.plot(range(1, len(model_dict["val_losses"])+1), model_dict["val_losses"], label='Validation Loss', marker='o', markersize=3, linewidth=2)
        # ax.set_ylim(0, None)
        ax.axvline(x=model_dict["best_epoch"], color='purple', linestyle='dotted', linewidth=1, label=f'Lowest val. loss')
        ax.set_title(f"lead time = {model_dict['lead_time']} month(s)", fontsize=9)
        ax.tick_params(axis='x', which='both', labelbottom=True) # ensure the x-axis labels are visible for each subplot
        ax.grid(linestyle='dashed')
        ax.legend(fontsize=8)
        # ax.set_xticks(np.arange(1, num_epochs), minor=True)
        ax.set_xlabel(f'Epoch', fontsize=9)
        if lead_time % 3 == 0:
            ax.set_ylabel(f'MSE Loss', fontsize=9)
    
        # add 5% margins around the x limits
        x_lower_limit = 0
        x_upper_limit = num_epochs            
        x_lower_bound = x_lower_limit - 0.05 * x_upper_limit  
        x_upper_bound = x_upper_limit + 0.05 * x_upper_limit  
        ax.set_xlim(x_lower_bound, x_upper_bound)
    
    
        
        # add 5% margins around the y limits
        y_lower_limit = 0
        y_upper_limit = 2.5            
        # y_upper_limit = 1            
        y_lower_bound = y_lower_limit - 0.05 * y_upper_limit  
        y_upper_bound = y_upper_limit + 0.05 * y_upper_limit  
        ax.set_ylim(y_lower_bound, y_upper_bound)
    
        # handles, labels = ax.get_legend_handles_labels()
        
    # fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.2, 0.88))
    fig.suptitle(f'Learning curves accross various lead times for {target} index prediction task\n\n'
                f'{subtitle}',
                fontsize=10, fontweight="bold")
    
    # fig.text(0.5, 0.0, 'Epoch', ha='center', va='center', fontsize=9, fontweight="bold")
    # fig.text(0.00, 0.5,  'MSE Loss', ha='center', va='center', rotation='vertical', fontsize=9, fontweight="bold")
    plt.tight_layout(h_pad=2, w_pad=None, rect=[0, 0, 1, 0.98]) # (left, bottom, right, top), default: (0, 0, 1, 1)
    
    if save_img == True:
        if img_filename == None:
            img_filename = f"_model_learning_curves_{model_dict['experiment_name']}.jpg"
        fig.savefig(img_folder+img_filename, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    if save_img == True:
        print(f"Saved {img_filename} to disk")

    
def plot_learning_curves_finetune(model_results, target, num_epochs, train_loss_display="batch", rows_to_show=4, subtitle=None, save_img=False, img_filename=None, img_folder=None):
    
    fig, axs = plt.subplots(nrows=rows_to_show, ncols=3, figsize=(11, rows_to_show*2.5), sharex=True, sharey=True)

    # fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(12, 18), sharex=True, sharey=True)
    
    # Iterate over the linear regression models' results and their respective axes
    for i, (model_dict, ax) in enumerate(zip(model_results[target], axs.flat)):
    # for (lead_time, model_dict), ax in zip(model_results[target].items(), axs.flat):

        
        # match the epoch axis scale for the train history per batch
        n_train = len(model_dict['train_loss_history']) # nb epochs * nb of train batches per train_dataloader of training set
        t_train = model_dict['train_num_epochs'] * (np.arange(n_train) / n_train)
        
        n_tune = len(model_dict['tune_loss_history']) # nb epochs * nb of tune batches per train_dataloader of finetuning set
        t_tune = + model_dict['train_num_epochs'] + model_dict['tune_num_epochs'] * (np.arange(n_tune) / n_tune)
    
        t_concat = np.concatenate((t_train, t_tune))

        # Plot Training and Validation losses
        if train_loss_display == "batch":
            ax.plot(t_concat, model_dict["train_loss_history"] + model_dict["tune_loss_history"], label='Train Loss (per batch)', marker='o', markersize=2, linewidth=1)
        elif train_loss_display == "epoch":
            ax.plot(range(1, len(model_dict["combined_train_losses"])+1), model_dict["combined_train_losses"], label='Train Loss', marker='o', markersize=3)
        else:
            print(f"Unknown train_loss_display value: {train_loss_display}. Choose 'batch' or 'epoch'")
            return
            
        ax.plot(range(1, len(model_dict["combined_val_losses"])+1), model_dict["combined_val_losses"], label='Validation Loss', marker='o', markersize=3, linewidth=2)
        # ax.set_ylim(0, None)
        ax.axvline(x=model_dict["train_best_epoch"], color='purple', linestyle='dotted', linewidth=1)
        ax.axvline(x=model_dict["train_num_epochs"] + model_dict["tune_best_epoch"], color='purple', linestyle='dotted', linewidth=1)
        # print(model_dict['transition_point'])
        ax.axvline(x=model_dict['transition_point'], color='r', linestyle='--', label='Fine-Tuning Start')

        ax.set_title(f"lead time = {model_dict['lead_time']} month(s)", fontsize=9)
        ax.tick_params(axis='x', which='both', labelbottom=True) # ensure the x-axis labels are visible for each subplot
        ax.grid(linestyle='dashed')
        ax.legend(fontsize=8)
        # ax.set_xticks(np.arange(1, num_epochs), minor=True)
        ax.set_xlabel(f'Epoch', fontsize=9)
        if i % 3 == 0:
            ax.set_ylabel(f'MSE Loss', fontsize=9)
    
        # add 5% margins around the x limits
        x_lower_limit = 0
        x_upper_limit = num_epochs            
        x_lower_bound = x_lower_limit - 0.05 * x_upper_limit  
        x_upper_bound = x_upper_limit + 0.05 * x_upper_limit  
        ax.set_xlim(x_lower_bound, x_upper_bound)
    
    
        
        # add 5% margins around the y limits
        y_lower_limit = 0
        y_upper_limit = 2.5            
        # y_upper_limit = 1            
        y_lower_bound = y_lower_limit - 0.05 * y_upper_limit  
        y_upper_bound = y_upper_limit + 0.05 * y_upper_limit  
        ax.set_ylim(y_lower_bound, y_upper_bound)
    
        # handles, labels = ax.get_legend_handles_labels()
        
    # fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.2, 0.88))
    fig.suptitle(f'Learning curves accross various lead times for {target} index prediction task\n\n'
                f'{subtitle}',
                fontsize=10, fontweight="bold")
    
    # fig.text(0.5, 0.0, 'Epoch', ha='center', va='center', fontsize=9, fontweight="bold")
    # fig.text(0.00, 0.5,  'MSE Loss', ha='center', va='center', rotation='vertical', fontsize=9, fontweight="bold")
    plt.tight_layout(h_pad=2, w_pad=None, rect=[0, 0, 1, 0.98]) # (left, bottom, right, top), default: (0, 0, 1, 1)
    
    if save_img == True:
        if img_filename == None:
            img_filename = f"_model_learning_curves_{model_dict['experiment_name']}.jpg"
        fig.savefig(img_folder+img_filename, dpi=300, bbox_inches='tight')
    
    
    plt.show()
    
    if save_img == True:
        print(f"Saved {img_filename} to disk")
    
    
def plot_pred_vs_actual_timeseries(model_results, target, abs_v_max, rows_to_show=4, subtitle=None, save_img=False, img_filename=None, img_folder=None):

    fig, axs = plt.subplots(nrows=rows_to_show, ncols=3, figsize=(11, rows_to_show*2), sharex=True, sharey=True)
    
    # Iterate over the linear regression models' results and their respective axes
    for i, (model_dict, ax) in enumerate(zip(model_results[target], axs.flat)):
    # for (lead_time, model_dict), ax in zip(model_results[target].items(), axs.flat):
    
        # Align timestamps of predictions and actual values
        predictions = pd.Series(model_dict['predictions'], index=model_dict['y_test'].index)
        predictions = predictions.sort_index()
        y_test = model_dict['y_test'].sort_index(ascending=True)
    
        # Evaluate the model on the validation data
        corr, _ = scipy.stats.pearsonr(predictions, y_test)
        mse = mean_squared_error(y_test, predictions)
        
        # Plot machine learning predictions and actual values on the same plot
        ax.plot(y_test, label=f'Actual {target.capitalize()} index values')
        ax.plot(predictions, '--', label='GCN Predictions')

        ax.set_ylim(-abs_v_max, abs_v_max)
        ax.yaxis.set_major_locator(MultipleLocator(1.5))  # Set y-ticks at every 1.5 increment
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=4))  # base=2 for every other year    
        ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))  # base=2 for every other year    
        ax.set_title(f"lead time = {model_dict['lead_time']} month(s)\nCorr: {corr:.2f}, MSE: {mse:.2f}", fontsize=9)
        ax.tick_params(axis='x', which='both', labelbottom=True, labelsize=9) # ensure the x-axis labels are visible for each subplot
        ax.grid(linestyle='dashed')
        
        if i % 3 == 0:
            ax.set_ylabel(f'{target.capitalize()} Index', fontsize=9)

        handles, labels = ax.get_legend_handles_labels()
        
    # fig.legend(handles, [labels[0], "GCN Predictions"], loc='upper center', bbox_to_anchor=(0.5, 0.959), ncol=2, shadow=True)
    fig.legend(handles, [labels[0], "GCN Predictions"], loc='upper center', bbox_to_anchor=(0.5, 0.84), ncol=2, shadow=True)
    fig.suptitle(f'Predicted vs. Actual {target.capitalize()} Index across various lead times'
                 f'\n\n{subtitle}',
                 fontsize=10, fontweight="bold")
    
    fig.text(0.5, 0.0, 'Years', ha='center', va='center', fontsize=9, fontweight="bold")
    # fig.text(0.00, 0.5,  f'{target.upper()}', ha='center', va='center', rotation='vertical', fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=2, w_pad=2, rect=[0, 0, 1, 0.96]) # (left, bottom, right, top), default: (0, 0, 1, 1)
        
    if save_img == True:
        if img_filename == None:
            img_filename = f"_model_pred_true_timeseries_{model_dict['experiment_name']}.jpg"
        fig.savefig(img_folder+img_filename, dpi=300, bbox_inches='tight')
        
    plt.show()
    
    if save_img == True:
        print(f"Saved {img_filename} to disk")



def plot_selected_pred_vs_actual_timeseries(model_results, target, abs_v_max, selected_lead_times, subtitle=None, individual_titles=True, save_img=False, img_filename=None, img_folder=None):
    rows_to_show = 1  # Since you want to show 3 lead times in 1 row
    cols_to_show = len(selected_lead_times)  # Number of columns will be equal to the number of selected lead times

    if subtitle is None and individual_titles is False:
        fig_height = 2 * rows_to_show
    if subtitle is None and individual_titles is True:
        fig_height = 2 * rows_to_show + 0.5
    else:
        fig_height = 2 * rows_to_show + 2 

    fig, axs = plt.subplots(nrows=rows_to_show, ncols=cols_to_show, figsize=(11, fig_height), sharex=True, sharey=True)
    
    # Ensure axs is always iterable, even if there's only one row/column
    if cols_to_show == 1:
        axs = [axs]

    # Filter the model results to include only the selected lead times
    filtered_model_results = [model_dict for model_dict in model_results[target] if model_dict['lead_time'] in selected_lead_times]

    # Iterate over the filtered model results and their respective axes
    for i, (model_dict, ax) in enumerate(zip(filtered_model_results, axs)):
        
        # Align timestamps of predictions and actual values
        predictions = pd.Series(model_dict['predictions'], index=model_dict['y_test'].index)
        predictions = predictions.sort_index()
        y_test = model_dict['y_test'].sort_index(ascending=True)
    
        # Evaluate the model on the validation data
        corr, _ = scipy.stats.pearsonr(predictions, y_test)
        mse = mean_squared_error(y_test, predictions)
        
        # Plot machine learning predictions and actual values on the same plot
        ax.plot(y_test, label=f'GODAS {target.capitalize()} index')
        ax.plot(predictions, '--', label='GCN Forecast')

        ax.set_ylim(-abs_v_max, abs_v_max)
        ax.yaxis.set_major_locator(MultipleLocator(1.5))  # Set y-ticks at every 1.5 increment
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=8))  # base=2 for every other year    
        ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))  # base=2 for every other year    
        
        if individual_titles:
            ax.set_title(f"lead time = {model_dict['lead_time']} month(s)\nCorr: {corr:.3f}, MSE: {mse:.3f}", fontsize=9)
        ax.tick_params(axis='x', which='both', labelbottom=True, labelsize=9) # ensure the x-axis labels are visible for each subplot
        ax.grid(linestyle='dashed')
        
        if i % cols_to_show == 0:
            ax.set_ylabel(f'{target.capitalize()} Index', fontsize=9)

        # handles, labels = ax.get_legend_handles_labels()
        ax.legend(loc='best', fontsize=8)

    if subtitle is not None:
        # fig.legend(handles, [labels[0], "GCN Predictions"], loc='upper center', bbox_to_anchor=(0.5, 0.84), ncol=2, shadow=True)
        fig.suptitle(f'Predicted vs. Actual {target.capitalize()} Index across selected lead times'
                    f'\n\n{subtitle}',
                    fontsize=10, fontweight="bold")
        
    
    fig.text(0.5, 0.0, 'Time', ha='center', va='center', fontsize=9, fontweight="normal")
    fig.tight_layout(h_pad=2, w_pad=2, rect=[0, 0, 1, 0.96]) # (left, bottom, right, top), default: (0, 0, 1, 1)
        
    if save_img == True:
        if img_filename is None:
            img_filename = f"_model_pred_true_timeseries_{model_dict['experiment_name']}.jpg"
        fig.savefig(img_folder + img_filename, dpi=300, bbox_inches='tight')
        
    plt.show()
    
    if save_img == True:
        print(f"Saved {img_filename} to disk")




def compute_time_interval(start_date_str, end_date_str):
    """
    Compute the time interval between two dates
    """
    # Convert the string dates to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Adjust the dates by adding one day to include the entire last day
    if end_date.day == 29 or end_date.day == 30 or end_date.day == 31:
        end_date += timedelta(days=1)
        
    if start_date.day == 29 or start_date.day == 30 or start_date.day == 31:
        start_date += timedelta(days=1)


    # Calculate the difference using relativedelta
    difference = relativedelta(end_date, start_date)


    # Extract the number of years, months, and days
    years = difference.years
    months = difference.months
    days = difference.days

    return years, months, days


    
    
def plot_corr_mse(correlation_dicts, mse_dicts, target, max_lead_time, mse_y_lim_max=None, corr_y_lim_min=-1, alpha_list=None, color_list=None, linestyle_list=None, linewidth_list=None, marker_list=None, legend_outside=False, suptitle=None, save_img=False, img_filename=None, img_folder=None):
    """
    Plots correlations and MSEs for a single target across various forecast models.
    """
    # Initialize lists if they are None
    if alpha_list is None:
        alpha_list = [1] * len(correlation_dicts)
    if color_list is None:
        color_list = [None] * len(correlation_dicts)
    if linestyle_list is None:
        linestyle_list = ['-'] * len(correlation_dicts)
    if linewidth_list is None:
        linewidth_list = [2] * len(correlation_dicts)  # Default to linewidth of 2
    if marker_list is None:
        marker_list = [None] * len(correlation_dicts)  # No markers by default
    
    if suptitle is None:
        fig_height = 4.5 + 2
    else:
        fig_height = 5.5 + 2
        
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9, fig_height))
    
    for i, ax in enumerate(axs.flat):
        if i == 0:
            for j, (label, data) in enumerate(correlation_dicts.items()):
                ax.plot(range(0, len(data[target])), data[target], label=label, 
                        alpha=alpha_list[j] if j < len(alpha_list) else 1, 
                        color=color_list[j] if j < len(color_list) else None,
                        linestyle=linestyle_list[j] if j < len(linestyle_list) else '-',
                        linewidth=linewidth_list[j] if j < len(linewidth_list) else 2,
                        marker=marker_list[j] if j < len(marker_list) else None)

            ax.set_ylabel('Pearson Correlation coef.', fontsize=9)
            ax.set_ylim(corr_y_lim_min, 1)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_title(f'Correlation between Forecasts and Actual {target} Indices Across Various Lead Times', fontweight="normal", fontsize=9)
            ax.set_yticks(np.arange(corr_y_lim_min, 1.25, 0.25))

        if i == 1:
            for j, (label, data) in enumerate(mse_dicts.items()):
                ax.plot(range(0, len(data[target])), data[target], label=label, 
                      alpha=alpha_list[j] if j < len(alpha_list) else 1, 
                        color=color_list[j] if j < len(color_list) else None,
                        linestyle=linestyle_list[j] if j < len(linestyle_list) else '-',
                        linewidth=linewidth_list[j] if j < len(linewidth_list) else 2,
                        marker=marker_list[j] if j < len(marker_list) else None)
            ax.set_ylabel('MSE', fontsize=9)
            ax.set_ylim(0, mse_y_lim_max)
            ax.set_title(f'Mean Square Errors between Forecasts and Actual {target} Indices Across Various Lead Times', fontweight="normal", fontsize=9)
            ax.set_xlabel('Lead time [months]', fontsize=9)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xticks(np.arange(0, max_lead_time * 5, 3))
        ax.set_xticks(np.arange(1, 49), minor=True)
        ax.set_xticklabels([], minor=True)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.margins(x=0)
        ax.grid(which='major', linewidth=0.75, alpha=0.5)
        ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.1)
        
        
        if legend_outside:
            ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
        else:
            ax.legend(loc='best', fontsize=9)
        
    
    if suptitle is not None:
        fig.suptitle(f'{suptitle}', fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=2)
    
    if save_img == True:
        if img_filename == None:
            img_filename = f"_model_corr_mse_{target}.jpg"
        fig.savefig(img_folder+img_filename, dpi=300, bbox_inches='tight')
    
    plt.show()

    if save_img == True:
        print(f"Saved {img_filename} to disk")
        
        
        
def plot_and_compare_timeseries(target, model_results_dict, abs_v_max, selected_lead_times, selected_models, suptitle=None, save_img=None, img_filename=None, img_folder=None):
    """
    Plot the selected lead times for the specified models.
    """
    print(f"Target: {target} index")
    
    # Filter the results based on selected lead times
    filtered_results = {model: [d for d in model_results_dict[model][target] if d['lead_time'] in selected_lead_times]
                        for model in selected_models}
    
    nrows = len(next(iter(filtered_results.values())))  # All filtered results should have the same number of rows
    ncols = len(selected_models)
    
    if suptitle is None:
        fig_height = 2 * nrows + 1
    else:
        fig_height = 2 * nrows + 2 
    
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, fig_height), sharex=True, sharey=True)
    
    for row_idx, lead_time_results in enumerate(zip(*filtered_results.values())):
        for col_idx, (model_name, model_dict) in enumerate(zip(selected_models, lead_time_results)):
            ax = axs[row_idx, col_idx] if nrows > 1 else axs[col_idx]  # Handle single row case
            
            # Align timestamps of predictions and actual values
            predictions = pd.Series(model_dict['predictions'], index=model_dict['y_test'].index).sort_index()
            y_test = model_dict['y_test'].sort_index(ascending=True)
            
            # Plot machine learning predictions and actual values on the same plot
            ax.plot(y_test, label=f'GODAS {target.capitalize()} index')
            ax.plot(predictions, '--', label=f'{model_name} Forecast')
            
            ax.set_ylim(-abs_v_max, abs_v_max)

            ax.yaxis.set_major_locator(MultipleLocator(1.5))  # Set y-ticks at every 1.5 increment
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.xaxis.set_major_locator(mdates.YearLocator(base=8))  # base=2 for every other year    
            ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))  # base=2 for every other year    
         
            ax.set_title(f"{model_name} (lead={model_dict['lead_time']} month(s))\n"
                            f"Corr: {model_dict['corr']:.3f}, MSE: {model_dict['mse']:.3f}", fontsize=9)

            # ax.set_title(f"lead time = {model_dict['lead_time']} month(s)\nCorr: {model_dict['corr']:.2f}, MSE: {model_dict['mse']:.2f}", fontsize=9)
            ax.tick_params(axis='x', which='both', labelbottom=True, labelsize=9) # ensure the x-axis labels are visible for each subplot
            ax.grid(linestyle='dashed')
            
            if col_idx == 0:
                ax.set_ylabel(f'{target.capitalize()} Index', fontsize=9)

            # handles, labels = ax.get_legend_handles_labels()
            ax.legend(loc='best', fontsize=7)
            
            
            # handles, labels = ax.get_legend_handles_labels()
    
    # fig.legend(handles, [labels[0], "Predictions"], loc='upper center', bbox_to_anchor=(0.5, 0.923), ncol=2, shadow=True)
    # fig.suptitle(f'Comparison of Models in Predicting the {target.capitalize()} Index Across Various Lead Times\n\n'
    #              f'Models trained on {model_dict["test_source_id"]} SST anomalies ({train_start_year}-{train_end_year})\n'
    #              f'GNN {gnn_subtitle}', fontsize=10, fontweight="bold")
    # fig.text(0.5, 0.0, 'Years', ha='center', va='center', fontsize=9, fontweight="bold")
    # fig.tight_layout(h_pad=2, w_pad=None, rect=[0, 0, 1, 0.97])
    
    if suptitle is not None:
        fig.suptitle(f'{suptitle}', fontsize=9, fontweight="bold")
    fig.tight_layout(h_pad=2)
    
    if save_img == True:
        if img_filename == None:
            img_filename = f"_compare_ts_{model_results_dict[selected_models[-1]][target][-1]['experiment_name']}.jpg"
        fig.savefig(img_folder+img_filename, dpi=300, bbox_inches='tight')
    
    plt.show()

    if save_img == True:
        print(f"Saved {img_filename} to disk")
    


