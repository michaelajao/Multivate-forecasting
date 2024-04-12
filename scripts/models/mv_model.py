%cd ../..

import os
import shutil
import numpy as np
import pandas as pd

# import plottnig libraries
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as goa

from pathlib import Path

# import libraries from src
from src.utils import plotting_utils, ts_utils
from src.forecasting.ml_forecasting import calculate_metrics

from tqdm import tqdm

np.random.seed(42)
tqdm.pandas()

os.makedirs('reports/results', exist_ok=True)

# Load the data 
preprocessed = Path('data/processed')
output = Path('reports/results')  

from collections import namedtuple

FeatureConfig = namedtuple(
    'FeatureConfig',
    [
        'target',
        'index_cols'
        'time_varying_cols',

    ],
)

train_df = pd.read_pickle(preprocessed / 'East of England_train.pkl')
val_df = pd.read_pickle(preprocessed / 'East of England_val.pkl')



# define the different feature configurations
# feature_configs = {
#     target = 'covidOccupiedMVBeds',
#     index_cols = ['date'],
    
# }

cols = list(train_df.columns)
target = 'covidOccupiedMVBeds'

# define the model