import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import cycle
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
from tqdm.autonotebook import tqdm
from src.utils import plotting_utils
from src.transforms.target_transformations import AutoStationaryTransformer
import plotly.io as pio
import warnings
import logging

# Description: This script contains the code for the second experiment in the project, 
# forecasting COVID-19 MVBeds using various RNN models and hyperparameter tuning with Simulated Annealing.

# Set seeds for reproducibility
pl.seed_everything(42)
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.set_float32_matmul_precision('high')

# Set default plotly template
pio.templates.default = "plotly_white"

# Ignore warnings
warnings.filterwarnings("ignore")

# Set logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and Prepare Data
data_path = Path("data/processed/merged_nhs_covid_data.csv")
data = pd.read_csv(data_path).drop("Unnamed: 0", axis=1)
data["date"] = pd.to_datetime(data["date"])

# Select a different area name
selected_area = "South West"
data_filtered = data[data["areaName"] == selected_area]

# Data Processing
data_filtered["date"] = pd.to_datetime(data_filtered["date"])
data_filtered.sort_values(by=["date", "areaName"], inplace=True)
data_filtered.drop(
    [
        "areaName",
        "areaCode",
        "cumAdmissions",
        "cumulative_confirmed",
        "cumulative_deceased",
        "population",
        "latitude",
        "longitude",
        "epi_week",
    ],
    axis=1,
    inplace=True,
)

def add_rolling_features(df, window_size, columns, agg_funcs=None):
    if agg_funcs is None:
        agg_funcs = ["mean"]
    added_features = {}
    for column in columns:
        for func in agg_funcs:
            roll_col_name = f"{column}_rolling_{window_size}_{func}"
            df[roll_col_name] = df[column].rolling(window_size).agg(func)
            if column not in added_features:
                added_features[column] = []
            added_features[column].append(roll_col_name)
    df.dropna(inplace=True)
    return df, added_features

# Configuration
window_size = 7
columns_to_roll = ["hospitalCases", "newAdmissions", "new_confirmed", "new_deceased"]
agg_funcs = ["mean", "std"]

# Apply rolling features for each column
data_filtered, added_features = add_rolling_features(
    data_filtered, window_size, columns_to_roll, agg_funcs
)

for column, features in added_features.items():
    print(f"{column}: {', '.join(features)}")

def add_lags(data, lags, features):
    added_features = []
    for feature in features:
        for lag in lags:
            new_feature = feature + f"_lag_{lag}"
            data[new_feature] = data[feature].shift(lag)
            added_features.append(new_feature)
    return data, added_features

lags = [1, 2, 3, 5, 7, 14, 21]
data_filtered, added_features = add_lags(data_filtered, lags, ["covidOccupiedMVBeds"])
data_filtered.dropna(inplace=True)

def create_temporal_features(df, date_column):
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["day_of_week"] = df[date_column].dt.dayofweek
    return df

data_filtered = create_temporal_features(data_filtered, "date")
data_filtered.set_index("date", inplace=True)

seird_data = pd.read_csv(f"reports/output/pinn_{selected_area}_output.csv")
seird_data["date"] = pd.to_datetime(seird_data["date"])
seird_data.set_index("date", inplace=True)

# Merge the two dataframes
merged_data = pd.merge(data_filtered, seird_data, left_index=True, right_index=True, how="inner")

# Set the target variable
target = "covidOccupiedMVBeds"
seasonal_period = 7
auto_stationary = AutoStationaryTransformer(seasonal_period=seasonal_period)

# Fit and transform the target column to make it stationary
data_stat = auto_stationary.fit_transform(merged_data[[target]], freq="D")
merged_data[target] = data_stat.values

merged_data.info()

# Get the minimum and maximum date from the data
min_date = merged_data.index.min()
max_date = merged_data.index.max()
date_range = max_date - min_date
print(f"Data ranges from {min_date} to {max_date} ({date_range.days} days)")

# Filter data between the specified dates
start_date = "2020-04-14"
end_date = "2020-12-30"
merged_data = merged_data[start_date:end_date]

# Split the data into training, validation, and testing sets
train_end = min_date + pd.Timedelta(days=date_range.days * 0.45)
val_end = train_end + pd.Timedelta(days=date_range.days * 0.15)
train = merged_data[merged_data.index <= train_end]
val = merged_data[(merged_data.index > train_end) & (merged_data.index < val_end)]
test = merged_data[merged_data.index > val_end]

total_sample = len(merged_data)
train_sample = len(train) / total_sample * 100
val_sample = len(val) / total_sample * 100
test_sample = len(test) / total_sample * 100

print(
    f"Train: {train_sample:.2f}%, Validation: {val_sample:.2f}%, Test: {test_sample:.2f}%"
)
print(
    f"Train: {len(train)} samples, Validation: {len(val)} samples, Test: {len(test)} samples"
)
print(
    f"Max date in train: {train.index.max()}, Min date in train: {train.index.min()}, Max date in val: {val.index.max()}, Min date in val: {val.index.min()}, Max date in test: {test.index.max()}, Min date in test: {test.index.min()}"
)

train_dates = (train.index.min(), train.index.max())
val_dates = (val.index.min(), val.index.max())
test_dates = (test.index.min(), test.index.max())

print(f"Train dates: {train_dates}, Val dates: {val_dates}, Test dates: {test_dates}")

