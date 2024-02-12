# %%
#Changing the working directory to the root
# %cd ../

# %%
import pandas as pd
import numpy as np


import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio
from itertools import cycle

import os
from pathlib import Path
import warnings
import matplotlib.dates as mdates
import torch

warnings.filterwarnings("ignore")

# %%
Path("data/")
# Check if a GPU is available
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# %%
data = pd.read_csv("../data/merged_data.csv")
data.head()

# %%
data["date"] = pd.to_datetime(data["date"])
data = data.drop(columns=["cumulativeCases", "cumAdmissions"])
data.head()

# %%
def format_plot(
    fig, legends=None, xlabel="Time", ylabel="Value", title="", font_size=15
):
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_layout(
        autosize=False,
        width=900,
        height=500,
        title_text=title,
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
        titlefont={"size": 20},
        legend_title=None,
        legend=dict(
            font=dict(size=font_size),
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=1,
        ),
        yaxis=dict(
            title_text=ylabel,
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
        xaxis=dict(
            title_text=xlabel,
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
        ),
    )
    return fig


def mase(actual, predicted, insample_actual):
    mae_insample = np.mean(np.abs(np.diff(insample_actual)))
    mae_outsample = np.mean(np.abs(actual - predicted))
    return mae_outsample / mae_insample


def forecast_bias(actual, predicted):
    return np.mean(predicted - actual)


def plot_forecast(
    pred_df, forecast_columns, forecast_display_names=None, save_path=None
):
    if forecast_display_names is None:
        forecast_display_names = forecast_columns
    else:
        assert len(forecast_columns) == len(forecast_display_names)

    mask = ~pred_df[forecast_columns[0]].isnull()
    colors = px.colors.qualitative.Set2  # Using a different color palette
    act_color = colors[0]
    colors = cycle(colors[1:])

    fig = go.Figure()

    # Actual data plot
    fig.add_trace(
        go.Scatter(
            x=pred_df[mask].index,
            y=pred_df[mask].covidOccupiedMVBeds,
            mode="lines",
            marker=dict(size=6, opacity=0.5),
            line=dict(color=act_color, width=2),
            name="Actual COVID-19 MVBeds trends",
        )
    )

    # Predicted data plot
    for col, display_col in zip(forecast_columns, forecast_display_names):
        fig.add_trace(
            go.Scatter(
                x=pred_df[mask].index,
                y=pred_df.loc[mask, col],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color=next(colors), width=2),
                name=display_col,
            )
        )

    return fig


def highlight_abs_min(s, props=""):
    return np.where(s == np.nanmin(np.abs(s.values)), props, "")

# %%
# Find the minimum and maximum dates
min_date = data["date"].min()
max_date = data["date"].max()

print("Minimum Date:", min_date)
print("Maximum Date:", max_date)

# Calculate the date ranges for train, val, and test sets
date_range = max_date - min_date
train_end = min_date + pd.Timedelta(days=date_range.days * 0.75)
val_end = train_end + pd.Timedelta(days=date_range.days * 0.10)

# Split the data into train, validation, and test sets based on the date ranges
train = data[data["date"] < train_end]
val = data[(data["date"] >= train_end) & (data["date"] < val_end)]
test = data[data["date"] >= val_end]

# Calculate the percentage of dates in each dataset
total_samples = len(data)
train_percentage = len(train) / total_samples * 100
val_percentage = len(val) / total_samples * 100
test_percentage = len(test) / total_samples * 100

print(
    f"# of Training samples: {len(train)} | # of Validation samples: {len(val)} | # of Test samples: {len(test)}"
)
print(
    f"Percentage of Dates in Train: {train_percentage:.2f}% | Percentage of Dates in Validation: {val_percentage:.2f}% | Percentage of Dates in Test: {test_percentage:.2f}%"
)
print(
    f"Max Date in Train: {train.date.max()} | Min Date in Validation: {val.date.min()} | Min Date in Test: {test.date.min()}"
)

# %%
train.info()

# %%
train.to_csv("../data/train.csv")
val.to_csv("../data/val.csv")
test.to_csv("../data/test.csv")

# %%
class LogTime:
    from time import time

    def __enter__(self):
        self.start_time = self.time()
        print("Starting operation...")

    def __exit__(self, type, value, traceback):
        elapsed_time = self.time() - self.start_time
        print(f"Operation completed in {elapsed_time} seconds.")


def add_lags(df, lags, column):
    added_features = []
    for lag in lags:
        lag_col_name = f"{column}_lag_{lag}"
        df[lag_col_name] = df[column].shift(lag)
        added_features.append(lag_col_name)
    # Drop rows with NaN values
    df.dropna(inplace=True)
    return df, added_features

# %%
lags = [1, 7]

data_with_lags, added_features = add_lags(data, lags, "covidOccupiedMVBeds")

# %%
data_with_lags.head()

# %%
def add_seasonal_rolling_features(df, rolls, seasonal_periods, columns, agg_funcs):
    added_features = []

    for column in columns:
        for roll in rolls:
            for period in seasonal_periods:
                for func in agg_funcs:
                    roll_column = f"{column}_roll_{roll}_period_{period}_{func}"

                    # Calculate the rolling feature
                    rolled = df[column].rolling(window=roll * period)
                    if func == "mean":
                        df[roll_column] = rolled.mean()
                    elif func == "std":
                        df[roll_column] = rolled.std()

                    added_features.append(roll_column)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    return df, added_features

# %%
# Define parameters for the function
rolls = [1]
seasonal_periods = [7]  # Example: 7 days for weekly, 30 days for monthly
columns = ["covidOccupiedMVBeds", "hospitalCases", "newAdmissions", "dailyCases"]
agg_funcs = ["mean"]

with LogTime():
    data, added_features = add_seasonal_rolling_features(
        data, rolls, seasonal_periods, columns, agg_funcs
    )

print(f"Features Created: {','.join(added_features)}")

# %%
data.head()
%cd ../
# %%
from src.feature_engineering.temporal_features import add_temporal_features

# Temporarily reset the index
data_reset = data.reset_index()

with LogTime():
    data_reset, added_features = add_temporal_features(
        data_reset,
        field_name="date",
        frequency="D",
        add_elapsed=True,
        drop=False,
        use_32_bit=True,
    )
print(f"Features Created: {','.join(added_features)}")

# Set the 'date' column back as the index
data = data_reset.set_index("date")

# %%
data.head()

# %%
from src.transforms.target_transformations import AutoStationaryTransformer

# Set the target variable
target = "covidOccupiedMVBeds"

# Initialize the AutoStationaryTransformer with a seasonality period
# Adjust the seasonal_period based on your data's seasonality (for example, 7 for weekly seasonality)
seasonal_period = 7
auto_stationary = AutoStationaryTransformer(seasonal_period=seasonal_period)

# Fit and transform the target column to make it stationary
data_stat = auto_stationary.fit_transform(data[[target]], freq="D")

# Replace the original target values with the transformed stationary values
data[target] = data_stat.values

# Print the transformed data to check
data.head()

# %%
data.info()

# %%
data.columns

# %%
# Get the minimum and maximum date from the data
min_date = data.index.min()
max_date = data.index.max()

# Calculate the range of dates
date_range = max_date - min_date

# Define the end date for the training set (75% of the data) and validation set (10% of the data)
train_end = min_date + pd.Timedelta(days=date_range.days * 0.75)
val_end = train_end + pd.Timedelta(days=date_range.days * 0.10)

# Split the data into train, validation, and test sets based on the date ranges
train = data[data.index < train_end]
val = data[(data.index >= train_end) & (data.index < val_end)]
test = data[data.index >= val_end]

# Calculate the percentage of dates in each dataset
total_samples = len(data)
train_percentage = len(train) / total_samples * 100
val_percentage = len(val) / total_samples * 100
test_percentage = len(test) / total_samples * 100

print(
    f"# of Training samples: {len(train)} | # of Validation samples: {len(val)} | # of Test samples: {len(test)}"
)
print(
    f"Percentage of Dates in Train: {train_percentage:.2f}% | Percentage of Dates in Validation: {val_percentage:.2f}% | Percentage of Dates in Test: {test_percentage:.2f}%"
)
print(
    f"Max Date in Train: {train.index.max()} | Min Date in Validation: {val.index.min()} | Min Date in Test: {test.index.min()}"
)

# %%
train.to_csv("../../Multivate-forecasting/data/model_data/targetTransf_train.csv")
val.to_csv("../data/model_data/targetTransf_val.csv")
test.to_csv("../data/model_data/targetTransf_test.csv")

# %%
# Concatenate the DataFrames
sample_df = pd.concat([train, val, test])

# Convert feature columns to float32
# Exclude the 'type' column from conversion as it's a string column
for col in sample_df.columns:
    if col != "type":
        sample_df[col] = sample_df[col].astype("float32")

sample_df.head()

# %%
columns_to_select = [
    "covidOccupiedMVBeds",
    "hospitalCases",
    "newAdmissions",
    "Vax_index",
    "dailyCases",
    "covidOccupiedMVBeds_lag_1",
    "covidOccupiedMVBeds_lag_7",
    "covidOccupiedMVBeds_roll_1_period_7_mean",
    "hospitalCases_roll_1_period_7_mean",
    "newAdmissions_roll_1_period_7_mean",
    "dailyCases_roll_1_period_7_mean",
    "_Month",
    "_WeekDay",
    "_Dayofweek",
    "_Dayofyear"
]

# %%
sample_df = sample_df[columns_to_select]
sample_df.head()

# %%
cols = list(sample_df.columns)
cols.remove("covidOccupiedMVBeds")
sample_df = sample_df[cols + ["covidOccupiedMVBeds"]]

# %%
target = "covidOccupiedMVBeds"
pred_df = pd.concat([train[[target]], val[[target]]])
sample_df.info()

# %%
import shutil
import random
from src.utils import plotting_utils
from src.dl.dataloaders import TimeSeriesDataModule
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from src.dl.multivariate_models import SingleStepRNNConfig, SingleStepRNNModel
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
from tqdm.notebook import tqdm
# For reproduceability set a random seed

torch.set_float32_matmul_precision("high")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
pl.seed_everything(42)

# %%
datamodule = TimeSeriesDataModule(
    data=sample_df,
    n_val=val.shape[0],
    n_test=test.shape[0],
    window=7,  # 7 days window
    horizon=1,  # single step
    normalize="global",  # normalizing the data
    batch_size=64,
    num_workers=0,
)
datamodule.setup()

# %%
# Check a few batches from the training dataloader
train_loader = datamodule.train_dataloader()
for x, y in train_loader:
    print("Input batch shape:", x.shape)
    print("Output batch shape:", y.shape)
    break

# %%
rnn_config = SingleStepRNNConfig(
    rnn_type="LSTM",
    input_size=15,  # 25 for multivariate time series
    hidden_size=128,
    num_layers=6,
    bidirectional=True,
    learning_rate=1e-3,
)
model = SingleStepRNNModel(rnn_config)
model.float()

# %%
trainer = pl.Trainer(
    devices=1,
    min_epochs=5,
    max_epochs=100,
    callbacks=[pl.callbacks.EarlyStopping(monitor="valid_loss", patience=3)],
)
trainer.fit(model, datamodule)
# Removing artifacts created during training
# shutil.rmtree("lightning_logs")

# %%
predictions = trainer.predict(model, datamodule.test_dataloader())
predictions = torch.cat(predictions).squeeze().detach().numpy()
# De-normalizing the predictions
predictions = predictions * datamodule.train.std + datamodule.train.mean

actuals = test["covidOccupiedMVBeds"].values

assert (
    actuals.shape == predictions.shape
), "Mismatch in shapes between actuals and predictions"

# %%
metric_record = []

# %%


# %%
algorithm_name = rnn_config.rnn_type

metrics = {
    "Algorithm": algorithm_name,
    "MAE": mae(actuals, predictions),
    "MSE": mse(actuals, predictions),
    "MASE": mase(actuals, predictions, train["covidOccupiedMVBeds"].values),
    "Forecast Bias": forecast_bias(actuals, predictions),
}

value_formats = ["{}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.2f}"]
metrics = {
    key: format_.format(value)
    for key, value, format_ in zip(metrics.keys(), metrics.values(), value_formats)
}

pred_df_ = pd.DataFrame({"Vanilla LSTM": predictions}, index=test.index)
pred_df = test.join(pred_df_)

metric_record.append(metrics)
print(metrics)

# %%
fig = plot_forecast(
    pred_df,
    forecast_columns=["Vanilla LSTM"],
    forecast_display_names=["Vanilla LSTM"],
)
title = f"Forecasting multivarate COVID-19 MVBeds with {algorithm_name}"
fig = format_plot(fig, title=title)
fig.update_xaxes(
    # type="date", range=["2023-01-01", "2023-04-01"], dtick="M1", tickformat="%b %Y"
)
save_path = f"images/forecast_multivarate_{algorithm_name}.png"
pio.write_image(fig, save_path)
fig.show()


# %%



