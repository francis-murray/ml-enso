#!/usr/bin/env python
# coding: utf-8

# # 8 ML-ENSO - GNNs

print("hello world")

import torch
import torch.optim as optim
torch.cuda.is_available()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}\n')

# Additional Info when using cuda
if device.type == 'cuda':
    print(f"• Torch cuda current device: {torch.cuda.current_device()}")
    print(f"• Torch cuda device count: {torch.cuda.device_count()}")
    print(f"• Torch cuda device name: {torch.cuda.get_device_name(0)}")



import os
import pandas as pd
import numpy as np
import xarray as xr
import utils as utils
import utilities.graph_utils as graph_utils
import scipy.stats
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates

from utilities.enso_dataset import ENSOGraphDataset
from utilities.gcn_model import GCN2LayerMeanPool, GCN2LayerConcat, GCN2LayerConcat2FCs, GCN3LayerConcat
import torch.nn as nn

from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader

from sklearn.covariance import graphical_lasso
from sklearn.covariance import GraphicalLasso
from sklearn.covariance import GraphicalLassoCV
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

from tqdm import tqdm

import json
import time
import networkx as nx
from datetime import datetime

from PIL import Image


from torchinfo import summary



from IPython.display import Image as IPImage, display


# # ============================================================

# ## Global variables

# CROSS VALIDATION SETTINGS
print()

# Longitudinal (time-series) split
num_folds = 3
tscv = TimeSeriesSplit(n_splits=num_folds, gap=12)
display(tscv)


param_grid = {
    'input_time_steps': [3], 
    'corr_coef_threshold': [0.80, 0.85, 0.90], # only used for Correlation method
    'learning_rate': [0.0001], 
    'weight_decay': [0], 
    'num_hidden_units': [100, 200, 250], # we keep the same number of hidden units for all layers
    'dropout_rate': [0, 0.5], 
    'use_batch_norm': [True, False] 
}


print("======= Parameter grid:")
print(param_grid)
print()


# In[55]:


param_list = list(ParameterGrid(param_grid))
print(f"Number of combinations of parameters: {len(param_list)}")
print(param_list)
print()


# ### Define folders

# In[8]:


RESULTS_FOLDER = 'saved_results'
RESULTS_ONI_FOLDER = 'saved_results/oni'
RESULTS_NINO34_FOLDER = 'saved_results/nino34'
RESULTS_EC_FOLDER = 'saved_results/ec'

target_to_results_folder = {
    "oni": RESULTS_ONI_FOLDER,
    "nino34": RESULTS_NINO34_FOLDER,
    "E": RESULTS_EC_FOLDER,
    "C": RESULTS_EC_FOLDER
}
target_to_results_folder


# ### General settings

# In[9]:


GENERAL_MODE = "training" # or "training" , "view_results_only"
FINETUNE_ONLY = False
TARGETS = ["oni"] # or ["E", "C", "oni", "nino34"]
TARGETS_STR = ''.join(TARGETS)
# INPUT_TIME_STEPS = 3
START_LEAD_TIME = 0
MAX_LEAD_TIME = 25
LEAD_TIME_STEP = 1

SAVE_PLOTS_TO_DISK = True


# In[10]:


for target in TARGETS:
    parent_directory = target_to_results_folder[target]
    directory = os.path.join(parent_directory, "gnn")
    
    if os.path.isdir(directory):
        print(f"{directory} already exist")
    else:
        print(f"Create new directory for {directory}")
        os.makedirs(directory, exist_ok=True) 


# ### Training settings

# In[11]:


BATCH_SIZE = 128
NUM_EPOCHS_TRAIN = 20
NUM_EPOCHS_FINETUNE = 20


# ### Graph settings

# In[12]:


NODES_SEL_METHOD = "Correlation" # "GLasso", "Correlation" or "Proximity"
NODES_TRIM_METHOD = "MostEdgesCC" # "LargestCC", "MostEdgesCC" or "MinDegree"
REMOVE_CONTINENTAL_NODES = True
FILLNA = False
INVALID_NODES_HANDLING = "complete_case" # or "mask"
CARTOGRAPHIC_PROJECTIONS = ["Orthographic"] # ["Orthographic", "Mollweide", "Robinson", "AlbersEqualArea", "Miller"]

DIFFERENT_GRAPH_PER_LEAD = False

if NODES_SEL_METHOD == "Proximity":
    SUBTITLE = f"Edges Selected Based on Proximity (One-Hop Neighbors)"
    SUBTITLE_SHORT = f"Proximity (1-Hop)"
    IMG_FILENAME_TAG = "proximity"
    SHRINKAGE = 0
elif NODES_SEL_METHOD == "Correlation":
    # CORR_COEF_THRESHOLD = 0.90
    # CORR_COEF_STR = f"{CORR_COEF_THRESHOLD:.2f}".replace('.',"")
    # SUBTITLE = f"Edges Selected Based on Correlation > {CORR_COEF_THRESHOLD}"
    SUBTITLE = f"Edges Selected Based on Correlation > CrossVal"
    # SUBTITLE_SHORT = f"Corr > {CORR_COEF_THRESHOLD}"
    SUBTITLE_SHORT = f"Corr > CrossVal"

    # IMG_FILENAME_TAG = f"CorrThr{CORR_COEF_STR}"
    IMG_FILENAME_TAG = f"CorrThrCrossVal"
    SHRINKAGE = 0
elif NODES_SEL_METHOD == "GLasso":
    # High alpha: sparser model (fewer edges). Might underfit data but simpler and more interpretable
    # Low alpha: denser model (more edges). Might overfit data but captures more conditional dependencies.
    GLASSO_ALPHA = 0.30
    GLASSO_ALPHA_STR = f"{GLASSO_ALPHA:.2f}".replace('.',"")
    # GLASSO_ALPHA_STR = "CV"
    SHRINKAGE = 0
    SUBTITLE = f"Edges Selected Using GLasso, alpha={GLASSO_ALPHA}, shrinkage={SHRINKAGE}"
    SHRINKAGE_STR = f"{SHRINKAGE:.2f}".replace('.',"")
    SUBTITLE_SHORT = f"GLasso (alpha={GLASSO_ALPHA}, shrinkage={SHRINKAGE})"
    SUBTITLE_SHORT = f"GLasso (alpha={GLASSO_ALPHA})"
    IMG_FILENAME_TAG = f"GLasso{GLASSO_ALPHA_STR}_Shrink{SHRINKAGE_STR}"
else:
    print(f"Invalid Nodes selection method")

SHRINKAGE_STR = f"{SHRINKAGE:.2f}".replace('.',"")

if NODES_TRIM_METHOD == "MinDegree":
    MIN_DEGREE = 0 # Set to 0 to keep all nodes, 1 to remove only isolated nodes
    NODES_TRIM_METHOD_STR = f"MinDegree{MIN_DEGREE}"
elif NODES_TRIM_METHOD == "LargestCC":
    NODES_TRIM_METHOD_STR = "LargestCC"
elif NODES_TRIM_METHOD == "MostEdgesCC":
    NODES_TRIM_METHOD_STR = "MostEdgesCC"
else:
    print(f"Invalid Nodes trim method")

print("SHRINKAGE_STR:  ", SHRINKAGE_STR)
print("NODES_SEL_METHOD:  ", NODES_SEL_METHOD)
print("NODES_TRIM_METHOD_STR: ", NODES_TRIM_METHOD_STR)
# print("SUBTITLE: ", SUBTITLE)
# print("SUBTITLE_SHORT: ", SUBTITLE_SHORT)
print("IMG_FILENAME_TAG: ", IMG_FILENAME_TAG)


# ### Model settings

# In[13]:


MODEL_TYPE = "GCN2LayerConcat"  # or GCN2LayerMeanPool, "GCN2LayerConcat", "GCN3LayerConcat", "GCN2LayerConcat2FCs",
# GCN_INPUT_FEATURES=INPUT_TIME_STEPS
# GCN_HIDDEN_DIM_1=250
# GCN_HIDDEN_DIM_2=250
GCN_HIDDEN_DIM_3=4
GCN_OUTPUT_DIM=1
ACTIVATION_FUNC = nn.Tanh() # nn.ReLU(), nn.LeakyReLU(), nn.Tanh(), nn.ELU()
ACTIVATION_FUNC_STR = str(ACTIVATION_FUNC).split("(")[0]
# USE_BATCH_NORM=False
# DROPOUT_RATE=0
# DROPOUT_RATE_STR = f"{DROPOUT_RATE:.2f}".replace('.',"")

# DROPOUT_RATE_STR

# MODEL_DETAILS = f"{MODEL_TYPE}_{GCN_HIDDEN_DIM_1}x{GCN_HIDDEN_DIM_2}_{ACTIVATION_FUNC_STR}_Bn{USE_BATCH_NORM}_p{DROPOUT_RATE_STR}"
# MODEL_DETAILS_TITLE = f"{MODEL_TYPE} (EmbDims={GCN_HIDDEN_DIM_1}x{GCN_HIDDEN_DIM_2}, Act={ACTIVATION_FUNC_STR}, BatchNorm={USE_BATCH_NORM}, Dropout={DROPOUT_RATE})"

# if MODEL_TYPE == "GCN3LayerConcat":
#     MODEL_DETAILS = f"{MODEL_TYPE}_{GCN_HIDDEN_DIM_1}x{GCN_HIDDEN_DIM_2}x{GCN_HIDDEN_DIM_3}"


# print("MODEL_DETAILS: ", MODEL_DETAILS)
# print("MODEL_DETAILS_TITLE: ", MODEL_DETAILS_TITLE)


# ### Hyperparam Grid Search Results

# In[14]:


# with open('gnn_cv_results_1.json', 'r') as file:
#     gnn_cv_results = json.load(file)


# In[15]:


# sorted_gnn_cv_results = sorted(gnn_cv_results, key=lambda row: row["avg_val_loss"])

# print("\nSorted Top 10 GNN CV results: ")
# for result_dict in sorted_gnn_cv_results[:10]:
#     print(f"• params: {str(result_dict['params']):>120} \tavg_val_loss: {result_dict['avg_val_loss']:.5f}")


# In[16]:


# hparams = sorted_gnn_cv_results[0]['params']
# hparams


# ### Optimizer settings

# In[17]:


OPTIMIZER_TYPE = "SGD" # "Adam" or "SGD"
if OPTIMIZER_TYPE == "Adam":
    pass
#     LEARNING_RATE = 1e-4
#     LEARNING_RATE_FINETUNING = 1e-5
#     WEIGHT_DECAY = 0 # 1e-4
#     OPTIMIZER_DETAILS = f"{OPTIMIZER_TYPE}_LR{LEARNING_RATE:.0e}_WD{WEIGHT_DECAY:.0e}"
#     OPTIMIZER_DETAILS_TITLE = f"{OPTIMIZER_TYPE} (LR={str(LEARNING_RATE)}, WD={str(WEIGHT_DECAY)})"

elif OPTIMIZER_TYPE == "SGD":
#     LEARNING_RATE = 1e-4
#     LEARNING_RATE_FINETUNING = 1e-5
#     WEIGHT_DECAY = 1e-6
    MOMENTUM = 0.8
    USE_NESTEROV = True
#     OPTIMIZER_DETAILS = f"{OPTIMIZER_TYPE}_LR{LEARNING_RATE:.0e}_WD{WEIGHT_DECAY:.0e}_Mom{MOMENTUM:.0e}_Nest{USE_NESTEROV}"
#     # OPTIMIZER_DETAILS_TITLE = f"{OPTIMIZER_TYPE} (LR={str(LEARNING_RATE)}/{str(LEARNING_RATE_FINETUNING)}, WD={str(WEIGHT_DECAY)}, Momentum={MOMENTUM:.2f}, Nesterov={USE_NESTEROV})"
#     OPTIMIZER_DETAILS_TITLE = f"{OPTIMIZER_TYPE} (LR={str(LEARNING_RATE)}, WD={str(WEIGHT_DECAY)}, Momentum={MOMENTUM:.2f}, Nesterov={USE_NESTEROV})"


print("OPTIMIZER_TYPE:  ", OPTIMIZER_TYPE)
# print("OPTIMIZER_DETAILS:  ", OPTIMIZER_DETAILS)
# print("OPTIMIZER_DETAILS_TITLE:  ", OPTIMIZER_DETAILS_TITLE)


# In[18]:


IMG_FOLDER = f'img/gnn_Target_{TARGETS_STR}_{IMG_FILENAME_TAG}/'
IMG_FOLDER


# In[19]:


if SAVE_PLOTS_TO_DISK == True:
    os.makedirs(IMG_FOLDER, exist_ok=True) 
    
IMG_FOLDER


# In[20]:


# target_to_img_folder = {
#     "oni": f'img/gnn_ONI_{IMG_FILENAME_TAG}/',
#     "nino34": f'img/gnn_Nino34_{IMG_FILENAME_TAG}/',
#     "E": f'img/gnn_E_{IMG_FILENAME_TAG}/',
#     "C": f'img/gnn_C_{IMG_FILENAME_TAG}/',
# }
# target_to_img_folder


# In[21]:


# if SAVE_PLOTS_TO_DISK == True:
#     for target in TARGETS:
#         directory = target_to_img_folder[target]
#         if os.path.isdir(directory):
#             print(f"{directory} already exist")
#         else:
#             print(f"Create new directory for {directory}")
#             os.makedirs(directory, exist_ok=True) 


# ### Select training region

# In[22]:


RESOLUTION = 5 # 1, 4 or 5 (in degrees)
FULL_MAP = False

if FULL_MAP == True:
    # Full map
    MIN_LAT, MAX_LAT = -90, +90
    MIN_LON, MAX_LON = 0, 360
    LAT_SLICE = None
    LON_SLICE = None
else:
    # Ham et. al: 0°–360°E, 55°S–60°N for three consecutive months 
    MIN_LAT, MAX_LAT = -55, 60
    MIN_LON, MAX_LON = 0, 360

    # Nino 3.4 region
    # MIN_LAT, MAX_LAT = -5, 5
    # MIN_LON, MAX_LON = 190, 240

    
    LAT_SLICE = slice(MIN_LAT, MAX_LAT)
    LON_SLICE = slice(MIN_LON, MAX_LON) 

    # Tello et. al: 0°–360°E, 75°S–65°N for three consecutive months 
    # MIN_LAT, MAX_LAT = -75, 65
    # MIN_LON, MAX_LON = 0, 360
    
    # LAT_SLICE = slice(MIN_LAT, MAX_LAT)
    # LON_SLICE = slice(MIN_LON, MAX_LON) 


# Niño 3.4 (5N-5S, 170W-120W)
# nino34_min_lat, nino34_max_lat = -5, 5
# nino34_min_lon, nino34_max_lon = 190, 240


# In[23]:


COORDINATES = utils.add_cardinals_title(MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)
COORDINATES


# ### Select dataset source 

# #### All Sources

# In[24]:


ALL_SOURCE_IDS = ["MRI-ESM2-0", "MIROC6", "BCC-ESM1", "FGOALS-f3-L", "GISS-E2-1-G", "HADISST"]
print(ALL_SOURCE_IDS)

ALL_SOURCE_IDS_STR = ', '.join(ALL_SOURCE_IDS)
print(ALL_SOURCE_IDS_STR)

ALL_SOURCE_IDS_FILENAME = '_'.join([source_id.split("-")[0] for source_id in ALL_SOURCE_IDS])
print(ALL_SOURCE_IDS_FILENAME)


# #### Train Sources

# In[25]:


# TRAIN_SOURCE_IDS = ["MRI-ESM2-0", "MIROC6", "BCC-ESM1", "FGOALS-f3-L", "GISS-E2-1-G"] # use all for ONI and Nino 34 prediction
TRAIN_SOURCE_IDS = ["MIROC6", "GISS-E2-1-G"] # use this subset for E and C index prediction
print(TRAIN_SOURCE_IDS)

TRAIN_SOURCE_IDS_STR = ', '.join(TRAIN_SOURCE_IDS)
print(TRAIN_SOURCE_IDS_STR)

TRAIN_SOURCE_IDS_FILENAME = '_'.join([source_id.split("-")[0] for source_id in TRAIN_SOURCE_IDS])
print(TRAIN_SOURCE_IDS_FILENAME)


# #### Test Sources

# In[26]:


# TEST_SOURCE_ID = "HADISST"
TEST_SOURCE_ID = "GODAS"
print(TEST_SOURCE_ID)

TEST_SOURCE_ID_FILENAME = TEST_SOURCE_ID.split("-")[0]
print(TEST_SOURCE_ID_FILENAME)


# #### Fine-tuning Sources

# In[27]:


FINETUNE_SOURCE_IDS = ["HADISST"]
print(FINETUNE_SOURCE_IDS)

FINETUNE_SOURCE_IDS_STR = ', '.join(FINETUNE_SOURCE_IDS)
print(FINETUNE_SOURCE_IDS_STR)


# ### Select dates for Train-Val-Test split
# Randomly splitting time series data into a train set and a test set is very risky. \
# In many climate datasets, time series have a non-neglible auto-correlation (correlation of a time series and its lagged version over time). \
# Think of it like this: the atmosphere usually has a "memory" of about 14 days, and the ocean roughly has a "memory" of about 2 years.
# 
# _(source: [CCAI Seasonal Forecasting tutorial](https://colab.research.google.com/drive/1eLEYFK3Mrae_nu1SzAjg7Sdf40bWnKTg#scrollTo=XrbMcDoscZM0&forceEdit=true&sandboxMode=true))_

# Ham et al split (Extended Data Table 2)
# - **Training dataset**:
#     - CMIP5 historical run: 1861-2004  
#     - Reanalysis (SODA): 1871-1973
# - **Validation dataset**:
#     - Reanalysis (GODAS): 1984-2017

# In[28]:


# # train_start_date = '1851-01-01'
# train_start_date = '1871-01-01'
# train_end_date = '1970-12-31'

# val_start_date = '1976-01-01'
# val_end_date =   '1990-12-31'

# test_start_date = '1996-01-01'
# test_end_date =   '2010-12-31'


# In[29]:


test_start_date_PRETRAIN = '1996-01-01'
test_end_date_PRETRAIN =   '2010-12-31'




# Following Ham et al split (Extended Data Table 2),
# and splitting the training dataset into train and val
# We include a 10 year gap betwen each set #

# train set: 83 years
train_start_date = '1871-01-01'
train_end_date = '1953-12-31'

# val set: 10 years
val_start_date = '1964-01-01'
val_end_date =   '1973-12-31'

# val set: 34 years
test_start_date = '1984-01-01'
test_end_date =   '2017-12-31'

print(f"Train start and end dates: {train_start_date} to {train_end_date}")
print(f"Val start and end dates: {val_start_date} to {val_end_date}")
print(f"Test start and end dates: {test_start_date} to {test_end_date}")


# In[30]:


train_years, train_months, train_days = graph_utils.compute_time_interval(train_start_date, train_end_date)
val_years, val_months, val_days = graph_utils.compute_time_interval(val_start_date, val_end_date)
test_years, test_months, test_days = graph_utils.compute_time_interval(test_start_date, test_end_date)
total_years = train_years + val_years + test_years
print(f"Total years (not including gaps between datsets): {train_years + val_years + test_years}")

print(f"The training period is   {train_years:>2} years, {train_months:>2} months, and {train_days:>2} days long ({train_years/total_years:.1%})")
print(f"The validation period is {val_years:>2} years, {val_months:>2} months, and {val_days:>2} days long ({val_years/total_years:.1%})")
print(f"The testing period is    {test_years:>2} years, {test_months:>2} months, and {test_days:>2} days long ({test_years/total_years:.1%})")
print()

years, months, days = graph_utils.compute_time_interval(train_end_date, val_start_date)
print(f"The interval between the training set and the validation set is:    {years:>2} years, {months:>2} months, and {days:>2} days long")

years, months, days = graph_utils.compute_time_interval(val_end_date, test_start_date)
print(f"The interval between the training set and the validation set is:    {years:>2} years, {months:>2} months, and {days:>2} days long")


# In[31]:


# extract year part of the dates
train_start_year = train_start_date.split("-")[0]
train_end_year   = train_end_date.split("-")[0]

val_start_year = val_start_date.split("-")[0]
val_end_year   = val_end_date.split("-")[0]

test_start_year = test_start_date.split("-")[0]
test_end_year   = test_end_date.split("-")[0]

print(f"Train start and end years: {train_start_year}-{train_end_year}")
print(f"Train start and end years: {val_start_year}-{val_end_year}")
print(f"Train start and end years: {test_start_year}-{test_end_year}")


# ### Experiment General Name

# In[32]:


# EXPERIMENT_GENERAL_NAME = (
#     f"{MODEL_DETAILS}_"
#     f"Train_{TRAIN_SOURCE_IDS_FILENAME}_"
#     f"Tune_{FINETUNE_SOURCE_IDS_STR}_"
#     f"{train_start_year}-{train_end_year}_"
#     f"Ep{NUM_EPOCHS_TRAIN:02d}-{NUM_EPOCHS_FINETUNE:02d}_"
#     f"Test_{TEST_SOURCE_ID_FILENAME}_"
#     f"{test_start_year}-{test_end_year}_"
#     f"Win{INPUT_TIME_STEPS:02d}_"
#     f"{utils.add_cardinals_fname(MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)}_"
#     f"Res{RESOLUTION}_"
#     # f"CovShrink{SHRINKAGE_STR}_"
#     f"{'GLasso' + GLASSO_ALPHA_STR if NODES_SEL_METHOD == 'GLasso' else 'CorrThr' + CORR_COEF_STR if NODES_SEL_METHOD == 'Correlation' else 'Proxim'}_"
#     f"{NODES_TRIM_METHOD_STR}_"
#     f"{OPTIMIZER_TYPE}"
#     # f"{OPTIMIZER_DETAILS}"
# )


# EXPERIMENT_GENERAL_NAME = EXPERIMENT_GENERAL_NAME.replace("MRI_MIROC6_BCC_FGOALS_GISS_HADISST", "ALL")
# EXPERIMENT_GENERAL_NAME = EXPERIMENT_GENERAL_NAME.replace("MRI_MIROC6_BCC_FGOALS_GISS", "CMIP6")
# print(EXPERIMENT_GENERAL_NAME)


# In[33]:


# EXPERIMENT_GENERAL_NAME.replace("GLassoCV", f"GLasso{TEST}")






# ## Training the GNN


# print(f"• GPU: {torch.cuda.get_device_name(0)}")
# print(f"• Targets: {TARGETS}")
# print(f"• Lead Times: {START_LEAD_TIME} to {MAX_LEAD_TIME}")
# # print(f"• Window size: {INPUT_TIME_STEPS}")
# print(f"• Nb Epochs: {NUM_EPOCHS_TRAIN}")
# print(f"• Batch Size: {BATCH_SIZE}")
# print(f"• Number of nodes: {G2.number_of_nodes()}")
# print(f"• Number of edges: {data.edge_index.shape[1]}")


# ### Pre-Training

#     # For each target
#         # For each lead time
#             ##### TRAIN MODEL #####
#             # Step 1: Assemble graph predictors predictands from multiple sources
#             # Step 2: Convert predictors to a time series node feature tensors
#             # Step 3: Create Train and Val ENSOGraphDatasets
#             # Step 4: Create batches of graphs using DataLoaders
#             # Step 5: Instantiate GCN model and set up an optimizer for it
#             # Step 6: Train model and save best model
#     
#             ##### EVALUATE MODEL #####
#             # Step 1: Assemble graph predictors predictands for Test set
#             # Step 2: Convert predictors to a time series node feature tensors
#             # Step 3: Create Test ENSOGraphDatasets
#             # Step 4: Create batches of graphs using DataLoaders
#             # Step 5: Load best saved model
#             # Step 6: Infer predictions and evaluate performance (Corr and MSE)
#             # Step 7: Store model performance metrics in a dictionary for the current lead time

# In[51]:


start_time_training = time.time()
start_time_training_datetime = datetime.fromtimestamp(start_time_training)
print("Start training time: ", start_time_training_datetime.strftime('%Y-%m-%d %H:%M:%S'))
gnn_training_results = {}


# In[52]:


if GENERAL_MODE != "view_results_only":
    if FINETUNE_ONLY != True:

        for target in TARGETS:
            print(f"\n\n===========================================================================================")
            print(f"Target to predict:         {target}")
            results_folder = target_to_results_folder.get(target)
            if results_folder is None:
                print(f"Unknown target {target}. Abort")
            save_results_folder = os.path.join(results_folder, "gnn")
        
            gnn_training_results[target] = []
            print(f"Target to predict:         {target}")
            print(f"• Training and validation sets: {TRAIN_SOURCE_IDS}")
            print(f"• Train start and end dates:  {train_start_date} to {train_end_date}")
            print(f"• Val start and end dates:    {val_start_date} to {val_end_date}")
            print(f"• Test start and end dates:   {test_start_date_PRETRAIN} to {test_end_date_PRETRAIN}")
            print()
            # for lead_time in range(START_LEAD_TIME, MAX_LEAD_TIME, LEAD_TIME_STEP):
            
            lead_time = 1 # perform the hyperparameter tuning only for lead time 1
            print(f"****************** Lead Time for hyper param tuning: {lead_time}")
            
            gnn_cv_results = []
            for params in tqdm(param_list, leave=True, desc="Params", disable=False):
                print(f"\n========================================================================================================================")
                print(f"========================================================================================================================")
                print(f"Combination of parameters:")
                print(params)


                # # ============================================================
                ##################################################################################################################
                ############################################ CONSTRUCT GRAPH FOR LAG 0 ###########################################
                ##################################################################################################################

                # ## Construct Graph
                lead_time_node_sel = 0
                print(f"CONSTRUCT GRAPH FOR LAG {lead_time_node_sel}:")
                print("\n################ TRAIN MODEL ################")

                # ### Construct Nodes

                # In[34]:



                # Step 1: Initialize lists to collect data from multiple sources
                X_list, y_list = [], []

                # Iterate over each source_id and assemble the data 
                # print(f"Assemble predictors and predictands from {ALL_SOURCE_IDS} + {TEST_SOURCE_ID} at lead time 1:")
                print(f"Assemble predictors and predictands from {TRAIN_SOURCE_IDS} + {FINETUNE_SOURCE_IDS} + {TEST_SOURCE_ID} at lead time 1:")
                for source_id in TRAIN_SOURCE_IDS + FINETUNE_SOURCE_IDS:
                    
                    X, y = graph_utils.assemble_graph_predictors_predictands(source_id, RESOLUTION, TARGETS[0], train_start_date, train_end_date, 
                                                                            lead_time=1, num_input_time_steps=params['input_time_steps'], 
                                                                            lat_slice=LAT_SLICE, lon_slice=LON_SLICE, 
                                                                            data_format="spatial", verbose=False)
                    print(f"• {source_id:<12}: X.shape: {X.shape}, y.shape: {y.shape}")

                    X_list.append(X)
                    y_list.append(y)


                # also append GODAS dataset to prevent nan values on graph (must be done separately as it has a different dates)
                X_test, y_test = graph_utils.assemble_graph_predictors_predictands(TEST_SOURCE_ID, RESOLUTION, TARGETS[0], test_start_date, test_end_date, 
                                                                        lead_time=1, num_input_time_steps=params['input_time_steps'], 
                                                                        lat_slice=LAT_SLICE, lon_slice=LON_SLICE, 
                                                                        data_format="spatial", verbose=False)
                print(f"• {TEST_SOURCE_ID:<12}: X_test.shape:   {X_test.shape}, y_test.shape:   {y_test.shape}")

                X_list.append(X_test)
                y_list.append(y_test)

                # Concatenate data from all sources
                X_concat = xr.concat(X_list, dim='time')
                y_concat = pd.concat(y_list)

                print(f"\nConcatenate data from all source ids:")
                print(f"• {'Concat':<12}: X_concat.shape: {X_concat.shape}, y_concat.shape: {y_concat.shape}")



                # In[35]:


                nb_lon_gridpoints = X_concat.shape[-1]
                nb_lat_gridpoints = X_concat.shape[-2]
                nb_lon_gridpoints, nb_lat_gridpoints


                # In[36]:


                # Generate the graph's Nodes with ids, positions and labels attributes based on gridded Xarray dataset, 
                G_nodes_only = graph_utils.construct_graph_nodes(X_concat)


                # In[37]:


                central_lon, central_lat = 200, 0  # Center of Niño 1+2: (275, -5)


                # In[38]:


                # title = "All nodes (no edges)"
                # graph_utils.plot_nodes(G_nodes_only, central_lon=central_lon, central_lat=central_lat, figsize=(6, 6), 
                #                       projection="Miller", title=title, save_img=False, 
                #                       img_filename=None, img_folder=IMG_FOLDER)


                # In[39]:


                # Create a positions dictionary of all nodes
                full_pos_dict = graph_utils.get_pos_dict(G_nodes_only)
                pos_array = torch.Tensor(list(full_pos_dict.values()))

                # reverse pos dict before removing isolated nodes
                full_reverse_pos_dict = graph_utils.get_reverse_pos_dict(G_nodes_only)

                all_nodes = list(G_nodes_only.nodes)


                # ### Construct Edges

                # #### Remove invalid / continents points

                # In[40]:


                # Identify and plot invalid (NaN and Zero variance) points
                nan_lonlat_indices, nan_lonlat_coords = graph_utils.get_nan_points(X_concat)
                zero_var_lon_lat_indices, zero_var_lonlat_coords = graph_utils.get_zero_variance_points(X_concat)
                invalid_lonlat_indices = np.concatenate((nan_lonlat_indices, zero_var_lon_lat_indices))
                invalid_lonlat_coords = np.concatenate((nan_lonlat_coords, zero_var_lonlat_coords))

                # Returns the indices that would sort an array
                index_array = np.argsort(invalid_lonlat_indices)

                # Sort merged NaN and Zero variance indices and coords
                sorted_invalid_lonlat_indices = invalid_lonlat_indices[index_array]
                sorted_invalid_lonlat_coords = invalid_lonlat_coords[index_array]


                # # Plot nodes with NaN values
                title = f"Nodes with NaN values accross all time points\n{TRAIN_SOURCE_IDS + FINETUNE_SOURCE_IDS + [TEST_SOURCE_ID]}"
                graph_utils.plot_coords(nan_lonlat_coords, figsize=(6, 6), title=title)

                # # Plot nodes with zero variance
                title = f"Nodes with zero variance accross all time points\n{TRAIN_SOURCE_IDS + FINETUNE_SOURCE_IDS + [TEST_SOURCE_ID]}"
                graph_utils.plot_coords(zero_var_lonlat_coords, figsize=(6, 6), title=title)

                # Plot all invalid nodes (NaN values and zero variance)
                title = f"Invalid Nodes\n{TRAIN_SOURCE_IDS + FINETUNE_SOURCE_IDS + [TEST_SOURCE_ID]}"
                graph_utils.plot_coords(sorted_invalid_lonlat_coords, figsize=(6, 6), title=title)




                #### Assemble data that will be used to generate the edges (Train only - ESM)

                # Step 1: Initialize lists to collect data from multiple sources
                X_train_list, y_train_list = [], []

                # Iterate over each source_id and assemble the data
                print(f"Assemble predictors and predictands from {TRAIN_SOURCE_IDS} at lead time 1:")
                for source_id in TRAIN_SOURCE_IDS:
                    X_train, y_train = graph_utils.assemble_graph_predictors_predictands(source_id, RESOLUTION, TARGETS[0], train_start_date, train_end_date, 
                                                                            lead_time=1, num_input_time_steps=params['input_time_steps'], 
                                                                            lat_slice=LAT_SLICE, lon_slice=LON_SLICE, 
                                                                            data_format="spatial", verbose=False)
                    print(f"• {source_id:<12}: X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")

                    X_train_list.append(X_train)
                    y_train_list.append(y_train)

                # Concatenate data from all sources
                X_train_all_source_ids = xr.concat(X_train_list, dim='time')
                y_train_all_source_ids = pd.concat(y_train_list)

                print(f"\nConcatenate data from all source ids:")
                print(f"• {'Concat':<12}: X_train.shape: {X_train_all_source_ids.shape}, y_train.shape: {y_train_all_source_ids.shape}")




                # Flatten longitude and latitudes
                X_train_flat = graph_utils.flatten_lonlat(X_train_all_source_ids, verbose=True)

                # Remove invalid values 
                X_train_flat_cleaned = graph_utils.handle_invalid_values(X_train_flat, sorted_invalid_lonlat_indices, method=INVALID_NODES_HANDLING, verbose=True)

                X_train_flat.shape
                X_train_flat_cleaned.shape

                total_number_nodes = X_train_flat.shape[-1]
                total_number_nodes


                # ##### Select nodes using Correlation, GLasso or Proximity

                # In[46]:


                start_time_edges_compute = time.time()

                # for lead_time_node_sel in range(START_LEAD_TIME, MAX_LEAD_TIME, LEAD_TIME_STEP):
                print(f"\n=============================\n")
                print(f"lead_time_node_sel = {lead_time_node_sel}\n")
                    
                # reset the graph with nodes only
                G = G_nodes_only.copy()


                if NODES_SEL_METHOD == "Correlation": 
                    # 1. compute correlation matrix for valid nodes
                    if INVALID_NODES_HANDLING == "mask":
                        # sst_corr_matrix_valid = graph_utils.compute_corr_matrix(X_train_flat_cleaned, masked_data=True)
                        raise Exception("'mask' method not implemented.")
                    else:
                        # sst_corr_matrix_valid = graph_utils.compute_corr_matrix_per_lead(X_train_flat_cleaned, lead_time_node_sel)
                        sst_corr_matrix_valid = graph_utils.compute_corr_matrix(X_train_flat_cleaned, masked_data=False)
                    print("\nValid correlation matrix's shape: ", sst_corr_matrix_valid.shape)
                    
                    # # Display Valid Correlation Matrix
                    # title = f"Valid Nodes Correlation Matrix\n(Pairwise Node Correlation at Lag {lead_time_node_sel})\n{TRAIN_SOURCE_IDS_STR}"
                    # filename = f"corr_matrix_valid_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel:02d}.jpg"
                    # graph_utils.display_corr_matrix(sst_corr_matrix_valid, title=title, save_img=SAVE_PLOTS_TO_DISK, img_filename=filename, img_folder=IMG_FOLDER)
                    
                    
                    # 2. Insert the valid correlation matrix into the full one
                    sst_corr_matrix_full = graph_utils.insert_valid_corr_into_full(sst_corr_matrix_valid, sorted_invalid_lonlat_indices, corr_matrix_full_order=total_number_nodes)
                    print("Full correlation matrix's shape: ", sst_corr_matrix_full.shape)
                    
                    
                    # # Display Full Correlation Matrix
                    # title = f"All Nodes Correlation Matrix\n(Pairwise Node Correlation at Lag {lead_time_node_sel})\n{TRAIN_SOURCE_IDS_STR}"
                    # filename = f"corr_matrix_full_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel:02d}.jpg"
                    # graph_utils.display_corr_matrix(sst_corr_matrix_full, title=title, save_img=SAVE_PLOTS_TO_DISK, img_filename=filename, img_folder=IMG_FOLDER)

                    # 3. Compute Full Ajacency matrix: a value above the threshold in the Correlation Matrix indicates an edge
                    Adj_matrix = graph_utils.compute_adj_matrix(sst_corr_matrix_full, params['corr_coef_threshold'])



                elif NODES_SEL_METHOD == "GLasso":
                    ####################################################################################################################################
                    ############# 1. Compute Covariance Matrix of valid nodes using sklearn function #############
                    sst_cov_matrix_valid_sklearn = graph_utils.compute_cov_matrix_sklearn(X_train_flat_cleaned, assume_centered=False)

                    

                    # # Display Valid Covariance Matrix
                    # title = f"Valid Nodes Covariance Matrix\n(Pairwise Node Covariance at Lag {lead_time_node_sel})\n{TRAIN_SOURCE_IDS_STR}"
                    # filename = f"cov_matrix_valid_sklearn_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel:02d}.jpg"
                    # graph_utils.display_cov_matrix(sst_cov_matrix_valid_sklearn, centered_cbar=False, 
                    #                               title=title, save_img=SAVE_PLOTS_TO_DISK, 
                    #                               img_filename=filename, img_folder=IMG_FOLDER)


                    ####################################################################################################################################
                    ############## 2. Compute the sparse inverse covariance (precision) matrix for valid nodes #############
                    print(f"Compute the sparse inverse covariance (precision) matrix with alpha={GLASSO_ALPHA} for valid nodes")
                    # _, prec_matrix_valid = graphical_lasso(sst_cov_matrix_valid_sklearn, alpha=GLASSO_ALPHA, mode='cd', tol=0.0001, enet_tol=0.0001, max_iter=100, verbose=True)
                    _, prec_matrix_valid = graphical_lasso(sst_cov_matrix_valid_sklearn, alpha=GLASSO_ALPHA, mode='cd', tol=0.001, enet_tol=0.0001, max_iter=100, verbose=True)
                    
                    print("prec_matrix_valid.shape: ", prec_matrix_valid.shape)
                    print("prec_matrix_valid\n", prec_matrix_valid)

                    
                    # title = f"Valid Nodes Lagged Precision Matrix\n({SUBTITLE_SHORT}, Lag {lead_time_node_sel})\n{TRAIN_SOURCE_IDS_STR}"
                    # filename = f"prec_matrix_valid_{IMG_FILENAME_TAG}_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel:02d}.jpg"
                    # graph_utils.display_cov_matrix(prec_matrix_valid, centered_cbar=False, title=title, save_img=SAVE_PLOTS_TO_DISK, img_filename=filename,img_folder=IMG_FOLDER)

                    ####################################################################################################################################
                    ############## 3. Compute Adjacency Matrix for valid nodes: a non-zero value in the precision matrix indicates an edge #############
                    lasso_adj_matrix_valid = (prec_matrix_valid != 0).astype(int)
                    
                    # title = f"Lasso Lagged Adjacency Matrix (Valid Nodes)\n({SUBTITLE_SHORT}, Lag {lead_time_node_sel})\n{TRAIN_SOURCE_IDS_STR}"
                    # filename = f"lasso_adj_matrix_valid_{IMG_FILENAME_TAG}_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel:02d}.jpg"
                    # graph_utils.display_adj_matrix(lasso_adj_matrix_valid, title)


                    ########################################################################################
                    ############## 4. Insert valid Adjacency Matrix into Full Adjacency matrix #############
                    lasso_adj_matrix_full = graph_utils.insert_valid_corr_into_full(lasso_adj_matrix_valid, sorted_invalid_lonlat_indices, corr_matrix_full_order=total_number_nodes)
                    Adj_matrix = lasso_adj_matrix_full.copy()



                elif NODES_SEL_METHOD == "Proximity":
                    # Generate Edges based on One Hop neighborhood
                    # G = graph_utils.construct_onehop_edges(G, 5)
                    G = graph_utils.construct_onehop_edges_with_lon_wrapping(G, 5)
                    
                    Adj_matrix = nx.adjacency_matrix(G)
                    
                    # convert the sparse matrix to a dense format
                    Adj_matrix = Adj_matrix.todense()
                    print(f"Proximity Adj_matrix.shape: {Adj_matrix.shape}")
                    
                    # Sparsity: for an undirected graph, the maximum possible number of edges is n*(n-1)/2
                    sparsity = (2 * G.number_of_edges()) / (G.number_of_nodes() * (G.number_of_nodes() - 1))
                    
                else:
                    print("Invalid Nodes Selection Method")




                # # Display Adjacency matrix
                # title = f"Full Adjacency Matrix (All Nodes)\n({SUBTITLE_SHORT}, Lag {lead_time_node_sel})\n{TRAIN_SOURCE_IDS_STR}"
                # filename = f"Adj_matrix_full_{IMG_FILENAME_TAG}_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel:02d}.jpg"
                # graph_utils.display_adj_matrix(Adj_matrix, title)


                # Display Adjacency Matrix with Coordinates on axes
                sparsity = graph_utils.compute_sparsity(Adj_matrix)
                # title = f"Binary Adjacency Matrix\n{SUBTITLE_SHORT}, Lag {lead_time_node_sel}, Sparsity={sparsity:.3f}\n{TRAIN_SOURCE_IDS_STR}"
                # filename = f"bin_adj_matrix_{IMG_FILENAME_TAG}_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel}"
                # graph_utils.display_adj_matrix_w_coords(G, Adj_matrix, full_pos_dict, title, save_img=SAVE_PLOTS_TO_DISK, img_filename=filename, img_folder=IMG_FOLDER)


                # Compute Edge List
                edge_list = graph_utils.compute_edge_list(Adj_matrix)
                print(f"Number of edges for Adjacency matrix ({SUBTITLE_SHORT}) for Lag {lead_time_node_sel}: {len(edge_list):,}")

                # Add Edges based on Adjacency Matrix
                G.add_edges_from(edge_list)

                print(f"\nNumber of nodes: {G.number_of_nodes():,}")
                print(f"Number of edges: {G.number_of_edges():,}")

                if G.number_of_edges() == 0:
                    print(f"No Edges for this lead time {lead_time_node_sel}. Skip")
                    # continue


                # ##### Trim Nodes and Edges

                # In[47]:

                ########################################################################################
                ############################ Trim Nodes and Edges ######################################
                print(f"Trim Nodes and Edge")
                if REMOVE_CONTINENTAL_NODES:
                    print("Remove Continentental nodes")
                    print(f"Number of invalid (continental) nodes: {len(invalid_lonlat_indices)} / {G.number_of_nodes()} ({len(invalid_lonlat_indices)/G.number_of_nodes():.2%})")
                    nodes_to_keep = [node for node in all_nodes if node not in invalid_lonlat_indices]

                    G_valid = graph_utils.remove_nodes_from(G, invalid_lonlat_indices)
                    G1 = G_valid.copy()

                    # title = f"{TRAIN_SOURCE_IDS_STR} Graph structure (valid nodes only) at Lag {lead_time_node_sel}\n({SUBTITLE})"
                    # img_filename = f"graph_structure_{IMG_FILENAME_TAG}_valid_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel}.jpg"
                else: 
                    print("Continentental nodes have NOT been removed")
                    nodes_to_keep = all_nodes
                    G1 = G.copy()
                    # title = f"{TRAIN_SOURCE_IDS_STR} Graph structure (ALL nodes) at Lag {lead_time_node_sel}\n({SUBTITLE})"
                    # img_filename = f"graph_structure_{IMG_FILENAME_TAG}_all_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel}.jpg"

                ### Remove Isolated Nodes from Graph ####
                print("Remove Isolated nodes (not connected to any nodes)")

                # identify isolated nodes (not connected to any nodes)
                isolated_nodes = graph_utils.get_isolated_nodes(G1)
                graph_utils.print_isolated_vs_connected_stats(G1)

                # remove isolated nodes from graph
                G_connected = graph_utils.remove_isolated_nodes(G1)
                data_connected = from_networkx(G_connected) 
                nb_nodes_connected = G_connected.number_of_nodes()
                nb_edges_connected = data_connected.edge_index.shape[1] # PyG counts undirected edges twice

                # draw connected nodes using cartopy (not connected to any nodes)
                # title = f"Connected nodes ({SUBTITLE_SHORT}) at Lag {lead_time_node_sel}\n({nb_nodes_connected} Nodes, {nb_edges_connected} DirEdges)\n{TRAIN_SOURCE_IDS_STR}"
                # img_filename = f"cartopy_graph_{IMG_FILENAME_TAG}_connected_nodes_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel}.jpg"
                # graph_utils.plot_nodes(G_connected, central_lon=central_lon, central_lat=central_lat, figsize=(6, 6), projection="Miller", title=title, save_img=SAVE_PLOTS_TO_DISK, img_filename=img_filename, img_folder=IMG_FOLDER)



                if NODES_TRIM_METHOD == "MinDegree":
                    print(f"Keep only nodes with a Minimum Degree of {MIN_DEGREE}")
                    low_degree_nodes = graph_utils.get_low_degree_nodes(G_connected, MIN_DEGREE)
                    G_min_degree = graph_utils.filter_nodes_by_min_degree(G_connected, MIN_DEGREE)
                    nodes_to_keep = list(G_min_degree.nodes())
                    print(f"Remaining nodes after filtering:  {G_min_degree.number_of_nodes():>4} / {G_connected.number_of_nodes():>4} ({G_min_degree.number_of_nodes()/G_connected.number_of_nodes():.2%})")

                    G2 = G_min_degree.copy()
                elif NODES_TRIM_METHOD=="LargestCC":
                    print(f"Keep only Largest Connected component")

                    largest_cc_nodes = graph_utils.get_largest_connected_component(G_connected)
                    
                    # Get largest connected components subgraph
                    G_largest_cc = G_connected.subgraph(largest_cc_nodes).copy()
                    nodes_to_keep = list(G_largest_cc.nodes())
                    G2 = G_largest_cc.copy()

                elif NODES_TRIM_METHOD=="MostEdgesCC":
                    print(f"Keep only Component with most edges")
                    most_edge_cc = graph_utils.get_component_with_most_edges(G_connected)

                    # Get components with the most edges' subgraph
                    G_most_edges = G_connected.subgraph(most_edge_cc).copy()
                    nodes_to_keep = list(G_most_edges.nodes())
                    G2 = G_most_edges.copy()
                else: 
                    print("Keep all nodes from previous step")
                    nodes_to_keep = all_nodes
                    G2 = G1.copy()

                data = from_networkx(G2)
                # print(data)

                nb_nodes = len(nodes_to_keep)
                nb_edges = data.edge_index.shape[1]


                # Plot Largest Connected Component or MinDegree cartopy
                central_lon, central_lat = 200, 0  # Center of Niño 1+2: (275, -5)        
                # title = f"{NODES_TRIM_METHOD_STR} ({SUBTITLE_SHORT}) at Lag {lead_time_node_sel}\n({nb_nodes} Nodes, {nb_edges} DirEdges)\n{TRAIN_SOURCE_IDS_STR}"
                # img_filename = f"cartopy_graph_{IMG_FILENAME_TAG}_{NODES_TRIM_METHOD_STR}_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel}.jpg"
                # graph_utils.plot_nodes(G2, central_lon=central_lon, central_lat=central_lat, figsize=(6, 6), projection="Miller", title=title, save_img=SAVE_PLOTS_TO_DISK, img_filename=img_filename, img_folder=IMG_FOLDER)

                # # Plot Largest Connected Component or MinDegree Edges
                # title = f"Graph structure ({NODES_TRIM_METHOD_STR} subgraph) at Lag {lead_time_node_sel}\n({nb_nodes} Nodes, {nb_edges} DirEdges)\n({SUBTITLE})\n{TRAIN_SOURCE_IDS_STR}"
                # img_filename = f"graph_structure_{IMG_FILENAME_TAG}_{NODES_TRIM_METHOD_STR}_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel}.jpg"
                # graph_utils.draw_graph_basic(G2, title, save_img=SAVE_PLOTS_TO_DISK, img_filename=img_filename, img_folder=IMG_FOLDER)


                # # Display Adjacency Matrix with Coordinates on axes
                # title = f"Binary Adjacency Matrix\n{SUBTITLE_SHORT}, Lag {lead_time_node_sel}, Sparsity={sparsity:.3f}\n({nb_nodes} Nodes, {nb_edges} DirEdges)\n{TRAIN_SOURCE_IDS_STR}"
                # filename = f"bin_adj_matrix_{IMG_FILENAME_TAG}_{TRAIN_SOURCE_IDS_FILENAME}_Lag{lead_time_node_sel}"
                # graph_utils.display_adj_matrix_w_coords(G2, Adj_matrix, full_pos_dict, title, save_img=SAVE_PLOTS_TO_DISK, img_filename=filename, img_folder=IMG_FOLDER)


                print()
                print(f"Number of Nodes: {G2.number_of_nodes()} / {G1.number_of_nodes()}")
                print(f"Number of edges: {data.edge_index.shape[1]}")
                # print(f"data.edge_index:\n{data.edge_index}")
                # print(len(nodes_to_keep))


                elapsed_time_edges_compute_seconds = time.time() - start_time_edges_compute
                elapsed_time_edges_compute_minutes = elapsed_time_edges_compute_seconds / 60
                print(f"\nElapsed time to compute edge list: {elapsed_time_edges_compute_minutes:.2f} minutes ({elapsed_time_edges_compute_seconds:.2f} seconds)")  

                ##################################################################################################################
                ######################################### END CONSTRUCT GRAPH FOR LAG 0 ###########################################
                ##################################################################################################################


















        
                ######################################################
                print("\n################ TRAIN MODEL ################")
                
                # if DIFFERENT_GRAPH_PER_LEAD:
                #     if lead_time not in graph_data_by_lead_time:
                #         print(f"No available data for lead time {lead_time}. Skip")
                #         continue
        
                #     data = graph_data_by_lead_time[lead_time]
                #     print(data)
                #     nodes_to_keep = nodes_to_keep_by_lead_time[lead_time]
                #     print(f"Number of nodes to keep: {len(nodes_to_keep)}")
        
                # experiment_name = (
                #     f"{EXPERIMENT_GENERAL_NAME}_"
                #     f"{target.capitalize()}Idx_"
                #     f"Lead{lead_time:02d}"
                # )
                # print(f"Experiment name: {experiment_name}")
        
                ##### Step 1: Assemble graph predictors predictands from multiple sources
                X_train_list, y_train_list, X_test_list = [], [], []
                X_val_list, y_val_list, y_test_list = [], [], []
        
                print(f"\nAssemble predictors and predictands from multiple sources:")

                for source_id in TRAIN_SOURCE_IDS:
                    X_train, y_train = graph_utils.assemble_graph_predictors_predictands(source_id, RESOLUTION, target, train_start_date, train_end_date, lead_time, 
                                                                                         num_input_time_steps=params['input_time_steps'], lat_slice=LAT_SLICE, lon_slice=LON_SLICE, verbose=False)
                    X_val, y_val = graph_utils.assemble_graph_predictors_predictands(source_id, RESOLUTION, target, val_start_date, val_end_date, lead_time, 
                                                                                     num_input_time_steps=params['input_time_steps'], lat_slice=LAT_SLICE, lon_slice=LON_SLICE, verbose=False)
                    # X_test, y_test = graph_utils.assemble_graph_predictors_predictands(source_id, RESOLUTION, target, test_start_date_PRETRAIN, test_end_date_PRETRAIN, lead_time, 
                    #                                                                    num_input_time_steps=params['input_time_steps'], lat_slice=LAT_SLICE, lon_slice=LON_SLICE, verbose=False)
        
                    print(f"• {source_id:<12}: X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}, X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")
        
                    X_train_list.append(X_train)
                    y_train_list.append(y_train)
                    X_val_list.append(X_val)
                    y_val_list.append(y_val)
                    # X_test_list.append(X_test)
                    # y_test_list.append(y_test)
        
                # Concatenate data from all sources
                X_train = xr.concat(X_train_list, dim='time')
                y_train = pd.concat(y_train_list)
                X_val = xr.concat(X_val_list, dim='time')
                y_val = pd.concat(y_val_list)
                # X_test = xr.concat(X_test_list, dim='time')
                # y_test = pd.concat(y_test_list)
        
        
                # print(f"\nConcatenate data from all sources:")
                print(f"• {'Concat':<12}: X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}, X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

        
        
                ##### Step 2: Convert predictors xarray.DataArrays to a time series node feature tensors ([1438, 3, total_number_nodes])
                train_node_features_ts = graph_utils.get_node_features_ts(X_train, fillna=FILLNA) 
                val_node_features_ts = graph_utils.get_node_features_ts(X_val, fillna=FILLNA)
        
        
                ##### Step 3: Create Train and Val ENSOGraphDatasets
                print(f"\n{'Train Dataset:':<15}", end="")
                train_dataset = ENSOGraphDataset(train_node_features_ts[:, :, nodes_to_keep], y_train, data.edge_index, pos_array[nodes_to_keep])
        
                print(f"{'Val Dataset:':<15}", end="")
                val_dataset = ENSOGraphDataset(val_node_features_ts[:, :, nodes_to_keep], y_val, data.edge_index, pos_array[nodes_to_keep])
        
        
                 # Longitudinal k-Fold Cross-Validation
                train_num_time_steps = X_train.shape[0]
                split_best_val_losses = []
                for split_num, (train_index, test_index) in tqdm(enumerate(tscv.split(np.arange(train_num_time_steps))), leave=True, desc="CV split", disable=False):
                    print(f"\nCross Val Split number {split_num}")
                    print(f"• TRAIN indices: First: {train_index[0]:>4}, Last: {train_index[-1]:>4}, Total: {len(train_index):>4}")
                    print(f"• VAL indices:   First: {test_index[0]:>4}, Last: {test_index[-1]:>4}, Total: {len(test_index):>4}")
                    
                train_subset = torch.utils.data.Subset(train_dataset, train_index)
                val_subset = torch.utils.data.Subset(train_dataset, test_index)
        
                ##### Step 4: Create batches of graphs using DataLoaders
                train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE)
                val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE)
        
        
                ##### Step 5: Instantiate GCN model and set up an optimizer for it
                # nb_nodes_per_graph = data.x.shape[1]
                nb_nodes_per_graph = len(nodes_to_keep)
                if MODEL_TYPE == "GCN2LayerConcat":
                    model = GCN2LayerConcat(params['input_time_steps'], nb_nodes_per_graph, params['num_hidden_units'], params['num_hidden_units'], GCN_OUTPUT_DIM, ACTIVATION_FUNC, dropout_rate=params['dropout_rate'], use_batch_norm=params['use_batch_norm'])
                elif MODEL_TYPE == "GCN3LayerConcat":
                    model = GCN3LayerConcat(params['input_time_steps'], nb_nodes_per_graph, params['num_hidden_units'], params['num_hidden_units'], GCN_HIDDEN_DIM_3, GCN_OUTPUT_DIM)
                elif MODEL_TYPE == "GCN2LayerConcat2FCs":
                    model = GCN2LayerConcat2FCs(params['input_time_steps'], nb_nodes_per_graph, params['num_hidden_units'], params['num_hidden_units'], GCN_OUTPUT_DIM)
                elif MODEL_TYPE == "GCN2LayerMeanPool":
                    model = GCN2LayerMeanPool(params['input_time_steps'], params['num_hidden_units'], params['num_hidden_units'], GCN_OUTPUT_DIM, dropout_rate=params['dropout_rate'], use_batch_norm=params['use_batch_norm'])
                else:
                    raise ValueError("Invalid model name. Use 'GCN2LayerConcat', 'GCN3LayerConcat', 'GCN2LayerConcat2FCs', or 'GCN2LayerMeanPool'.")
    
                if lead_time == 0:
                    print(f"\nModel Architecture: \n{model}")
        
        
                if OPTIMIZER_TYPE == "Adam":
                    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
                elif OPTIMIZER_TYPE == "SGD":
                    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=MOMENTUM, nesterov=USE_NESTEROV)
                else: 
                    raise ValueError("Invalid optimizer name. Use 'Adam' or 'SGD'")
    
                if lead_time == 0:
                    print(f"\nOptimizer: \n{optimizer}")
        
                
                ##### Step 6: train model and save best model
                train_losses, train_loss_history, val_losses, best_epoch = graph_utils.train_network(model, nn.MSELoss(), optimizer, train_dataloader, val_dataloader, 
                                                                                    experiment_name="GCN_cross_validation", target=target, num_epochs=NUM_EPOCHS_TRAIN, verbose=False, save_model=False)
                print("• Epoch of best saved model (best validation loss): ", best_epoch)
        
        
                print(f"val_losses: {val_losses}")
                best_val_loss = val_losses[best_epoch-1]
                print(f"• Best val loss: {best_val_loss} (epoch {best_epoch-1})")
                split_best_val_losses.append(best_val_loss)
                
                # ##########################################################
                # print("\n################ EVALUATE TRAINED MODEL ################")
                # print(f"Evaluate best saved model for lead time {lead_time} on {TRAIN_SOURCE_IDS} test set ({test_start_date_PRETRAIN} to {test_end_date_PRETRAIN}):")
        
                # # ##### Step 1: Assemble graph predictors predictands for Test set
                # # X_test, y_test = graph_utils.assemble_graph_predictors_predictands(TEST_SOURCE_ID, RESOLUTION, target, test_start_date, test_end_date, lead_time, num_input_time_steps=params['input_time_steps'], 
                # #                                                  lat_slice=LAT_SLICE, lon_slice=LON_SLICE, verbose=False)
        
                # ##### Step 2: Convert predictors xarray.DataArrays to a time series node feature tensors
                # test_node_features_ts = graph_utils.get_node_features_ts(X_test, fillna=FILLNA)
        
                # ##### Step 3: Create Test ENSOGraphDatasets
                # print(f"{'Test Dataset:':<15}", end="")
                # test_dataset = ENSOGraphDataset(test_node_features_ts[:, :, nodes_to_keep], y_test, data.edge_index, pos_array[nodes_to_keep])
        
                # ##### Step 4: Create batches of graphs using DataLoaders
                # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
                # ##### Step 5: Load best saved model
                # graph_utils.load_model(model, optimizer, "train_"+experiment_name, target, device, verbose=True)
                # display(summary(model))

                # ##### Step 6: Infer predictions and evaluate performance (Corr and MSE)
                # predictions = graph_utils.infer(model, test_dataloader, verbose=False)
        
                # # Check for inf or NaN in predictions
                # if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                #     print("NaN or inf found in predictions")
                #     # Optionally, print the indices of these values
                #     print("Indices of NaN in predictions:", np.where(np.isnan(predictions)))
                #     print("Indices of inf in predictions:", np.where(np.isinf(predictions)))
        
                # corr, _ = scipy.stats.pearsonr(predictions, y_test)
                # mse = mean_squared_error(y_test, predictions)
                # print(f"• Evaluation results on Test set using best saved model: corr: {corr:.3f}, mse: {mse:.3f}")
        
                ##### Step 7: Store model performance metrics in a dictionary for the current lead time
                # gnn_dict = {
                #     'experiment_name': "train_"+experiment_name,
                #     'train_source_ids': TRAIN_SOURCE_IDS,
                #     'test_source_id': TEST_SOURCE_ID,
                #     'nb_train_samples': X_train.shape[0],
                #     'nb_test_samples': X_test.shape[0],
                #     'target': target, 
                #     'train_start_date': train_start_date,
                #     'train_end_date': train_end_date,
                #     'val_start_date': val_start_date,
                #     'val_end_date': val_end_date,
                #     'test_start_date': test_start_date_PRETRAIN,
                #     'test_end_date': test_end_date_PRETRAIN,
                #     'lead_time': lead_time,
                #     'num_input_time_steps': params['input_time_steps'],
                #     'y_test': y_test,
                #     'predictions': predictions,
                #     'corr': corr, 
                #     'mse': mse,
                #     'train_losses': train_losses,
                #     'val_losses': val_losses,
                #     'train_loss_history': train_loss_history,
                #     'best_epoch': best_epoch,
                #     'num_epochs': NUM_EPOCHS_TRAIN
                # }
                
                gnn_cv_dict = { 
                    'target': target,
                    'lead_time' : lead_time,
                    'params': params,
                    'avg_val_loss': np.mean(split_best_val_losses),
                }
                gnn_cv_results.append(gnn_cv_dict)
        
                # gnn_training_results[target].append(gnn_dict)
                # print()
                
                
            filename = 'gnn_cv_results.json'
            filebase, fileext = os.path.splitext(filename)
            counter = 1

            # Check if file exists and modify the filename if it does
            while os.path.exists(filename):
                filename = f"{filebase}_{counter}{fileext}"
                counter += 1
                    
            with open(filename, 'w') as file:
                json.dump(gnn_cv_results, file)


            print("\nUnsorted GNN CV results: ")
            for result_dict in gnn_cv_results:
                print(f"• params: {str(result_dict['params']):>70} \tavg_val_loss: {result_dict['avg_val_loss']:.5f}")


            sorted_gnn_cv_results = sorted(gnn_cv_results, key=lambda row: row["avg_val_loss"])
            sorted_gnn_cv_results

            print("\nSorted GNN CV results: ")
            for result_dict in sorted_gnn_cv_results:
                print(f"• params: {str(result_dict['params']):>70} \tavg_val_loss: {result_dict['avg_val_loss']:.5f}")
                
        
            # gnn_results_filename = "results_train_" + experiment_name.split("_Lead")[0] + ".pkl"
        
            # # Saved GCN model results for each target to disk as a pickle file
            # filepath = os.path.join(save_results_folder, gnn_results_filename)
            # with open(filepath, "wb") as file:
            #     pickle.dump(gnn_training_results[target], file)
            #     print(f"Saved GCN trained model results for target {target} to \n• {filepath}")



    

if GENERAL_MODE != "view_results_only":
    if FINETUNE_ONLY != True:
        elapsed_time_training_seconds = time.time() - start_time_training
        hours_train, remainder_train = divmod(elapsed_time_training_seconds, 3600)
        minutes_train, seconds_train = divmod(remainder_train, 60)
        hhmmss_elapsed_train_time = f'{int(hours_train):02}:{int(minutes_train):02}:{int(seconds_train):02}'
        print(f"\nElapsed time to train the GNN: {hhmmss_elapsed_train_time}")  
        print(f"• GPU: {torch.cuda.get_device_name(0)}")
        print(f"• Targets: {TARGETS}")
        # print(f"• Lead Times: {START_LEAD_TIME} to {MAX_LEAD_TIME}")
        print(f"• Lead Time: {lead_time}")
        print(f"• Window size: {params['input_time_steps']}")
        print(f"• Nb Epochs: {NUM_EPOCHS_TRAIN}")
        print(f"• Batch Size: {BATCH_SIZE}")
        print(f"• Number of nodes: {G2.number_of_nodes()}")
        print(f"• Number of edges: {data.edge_index.shape[1]}")
        num_params = sum(p.numel() for p in model.parameters())
        print(f"• Total number of parameters: {num_params:,}")
        print(f"• Parameter details:")
        for p in model.parameters():
            print(f"    • {p.numel():,}")

