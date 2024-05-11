# Path to source code
%cd ../../

# Imports for handling data
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import cycle
# from sklearn.model_selection import TimeSeriesSplit, train_test_split

# Imports for machine learning
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
# from sklearn.linear_model import LinearRegression
# from scipy.stats import spearmanr

# Imports for visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# Progress bar
from tqdm.autonotebook import tqdm
# Enable progress apply for pandas
tqdm.pandas()


# Local imports for data loaders and models
from src.utils import plotting_utils
from src.dl.dataloaders import TimeSeriesDataModule
from src.dl.multivariate_models import SingleStepRNNConfig, SingleStepRNNModel
from src.transforms.target_transformations import AutoStationaryTransformer


pl.seed_everything(42)
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    
torch.set_float32_matmul_precision('high')

# Set default plotly template
import plotly.io as pio
pio.templates.default = "plotly_white"

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# %load_ext tensorboard

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
# Load and Prepare Data
data_path = Path("data/processed/merged_nhs_covid_data.csv")
data = pd.read_csv(data_path).drop("Unnamed: 0", axis=1)
data["date"] = pd.to_datetime(data["date"])
selected_area = "Midlands"  # "London", "South East", "North West", "East of England", "South West", "West Midlands", "East Midlands", "Yorkshire and The Humber", "North East"
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

# %%
def create_temporal_features(df, date_column):
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["day_of_week"] = df[date_column].dt.dayofweek
    return df


data_filtered = create_temporal_features(data_filtered, "date")
data_filtered["date"] = pd.to_datetime(data_filtered["date"])
data_filtered = data_filtered.set_index("date")

target = "covidOccupiedMVBeds"

seasonal_period = 7
auto_stationary = AutoStationaryTransformer(seasonal_period=seasonal_period)

# Fit and transform the target column to make it stationary
data_stat = auto_stationary.fit_transform(data_filtered[[target]], freq="D")

# Replace the original target values with the transformed stationary values
data_filtered[target] = data_stat.values

# Print the transformed data to check
data_filtered.head()

# %%
data_filtered.info()

# %%
# Get the minimum and maximum date from the data
min_date = data_filtered.index.min()
max_date = data_filtered.index.max()
# Calculate the range of dates
date_range = max_date - min_date
print(f"Data ranges from {min_date} to {max_date} ({date_range.days} days)")

# %%
# Filter data between the specified dates
start_date = "2020-04-14"
end_date = "2021-12-30"
data_filtered = data_filtered[start_date:end_date]

# %%
# selecting 1 year data for training and 2 months data for validation and 3 months data for testing
# Calculate the date ranges for train, val, and test sets
train_end = min_date + pd.Timedelta(days=date_range.days * 0.45)
val_end = train_end + pd.Timedelta(days=date_range.days * 0.15)

# Split the data into training, validation, and testing sets
train = data_filtered[data_filtered.index <= train_end]
val = data_filtered[(data_filtered.index > train_end) & (data_filtered.index < val_end)]
test = data_filtered[data_filtered.index > val_end]

# Calculate the percentage of dates in each dataset
total_sample = len(data_filtered)
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
sample_df = pd.concat([train, val, test])

# Convert all the feature columns to float32
for col in sample_df.columns:
    sample_df[col] = sample_df[col].astype("float32")
    
columns_to_select = [
    "covidOccupiedMVBeds",
    "covidOccupiedMVBeds_lag_1",
    "covidOccupiedMVBeds_lag_2",
    "covidOccupiedMVBeds_lag_3",
    "covidOccupiedMVBeds_lag_5",
    "covidOccupiedMVBeds_lag_7",
    "covidOccupiedMVBeds_lag_14",
    "covidOccupiedMVBeds_lag_21",
    "month",
    "day",
    "day_of_week",
]

sample_df = sample_df[columns_to_select]
cols = list(sample_df.columns)
cols.remove("covidOccupiedMVBeds")
sample_df = sample_df[cols + ["covidOccupiedMVBeds"]]
target = "covidOccupiedMVBeds"
pred_df = pd.concat([train[[target]], val[[target]]])

datamodule = TimeSeriesDataModule(
    data=sample_df,
    n_val=val.shape[0],
    n_test=test.shape[0],
    window=7,  # 7 days window
    horizon=1,  # single step
    normalize="global",  # normalizing the data
    batch_size=32,
    num_workers=0,
)
datamodule.setup()

# Check a few batches from the training dataloader
train_loader = datamodule.train_dataloader()
for x, y in train_loader:
    print("Input batch shape:", x.shape)
    print("Output batch shape:", y.shape)
    break

# RNN Model
rnn_config = SingleStepRNNConfig(
    rnn_type="RNN",
    input_size=11,  # 25 for multivariate time series
    hidden_size=32,  # hidden size of the RNN
    num_layers=5, # number of layers
    bidirectional=False, # bidirectional RNN
    learning_rate=1e-3,
)
model = SingleStepRNNModel(rnn_config)
# model.float()

trainer = pl.Trainer(
    # logger=logger,
    min_epochs=5,
    max_epochs=100,
    accelerator = "gpu",
    devices = 1,
    callbacks=[EarlyStopping(monitor="valid_loss", patience=10)],
)
trainer.fit(model, datamodule)

def process_predictions_and_metrics(model, datamodule, algorithm_name, tag, train, test, metric_record):
    # Predictions from the model
    predictions = trainer.predict(model, datamodule.test_dataloader())
    predictions = torch.cat(predictions).squeeze().detach().numpy()
    # De-normalizing the predictions
    predictions = predictions * datamodule.train.std + datamodule.train.mean

    # Actual values
    actuals = test["covidOccupiedMVBeds"].values
    assert actuals.shape == predictions.shape, "Mismatch in shapes between actuals and predictions"

    # Calculate metrics
    metrics = {
        "Algorithm": f"{algorithm_name} ({tag})",
        "MAE": mae(actuals, predictions),
        "MSE": mse(actuals, predictions),
        "MASE": mase(actuals, predictions, train["covidOccupiedMVBeds"].values),
        "Forecast Bias": forecast_bias(actuals, predictions),
    }

    # Format metrics
    value_formats = ["{}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.2f}"]
    metrics = {
        key: format_.format(value)
        for key, value, format_ in zip(metrics.keys(), metrics.values(), value_formats)
    }

    # Create DataFrame and append to the test data
    pred_df_ = pd.DataFrame({f"{tag} {algorithm_name}": predictions}, index=test.index)
    pred_df = test.join(pred_df_)

    # Append metrics to the record and print
    metric_record.append(metrics)
    print(metrics)
    
    return pred_df, metrics

# Plot the forecast
def create_and_save_forecast_plot(df, algorithm_name, experiment_type, start_date, end_date):
    forecast_column = f"{experiment_type} {algorithm_name}"
    forecast_display_name = forecast_column
    
    fig = plot_forecast(
        df,
        forecast_columns=[forecast_column],
        forecast_display_names=[forecast_display_name]
    )
    
    title = f"Forecasting COVID-19 MVBeds with {experiment_type} {algorithm_name}"
    fig = format_plot(fig, title=title)
    
    fig.update_xaxes(
        type="date",
        range=[start_date, end_date],
        dtick="M1",
        tickformat="%b %Y"
    )

    # save as PDF
    save_path = f"reports/images/forecast_{selected_area}_{experiment_type}_{algorithm_name}.pdf"
    pio.write_image(fig, save_path)
    fig.show()

metric_record = []

algorithm_name = rnn_config.rnn_type

pred_df, metrics_vanilla = process_predictions_and_metrics(
    model, datamodule, algorithm_name, "Vanilla", train, test, metric_record
)

# Create and save the forecast plot
create_and_save_forecast_plot(
    pred_df,
    algorithm_name,
    "Vanilla",
    "2021-09-26",
    "2021-12-31"
)

# LSTM Model
rnn_config = SingleStepRNNConfig(
    rnn_type="LSTM",
    input_size=11,  # 25 for multivariate time series
    hidden_size=32,  # hidden size of the RNN
    num_layers=5, # number of layers
    bidirectional=False, # bidirectional RNN
    learning_rate=1e-3,
)
model = SingleStepRNNModel(rnn_config)
# model.float()

trainer = pl.Trainer(
    # logger=logger,
    min_epochs=5,
    max_epochs=100,
    accelerator = "gpu",
    devices = 1,
    callbacks=[EarlyStopping(monitor="valid_loss", patience=10)],
)
trainer.fit(model, datamodule)

algorithm_name = rnn_config.rnn_type

pred_df, metrics_vanilla = process_predictions_and_metrics(
    model, datamodule, algorithm_name, "Vanilla", train, test, metric_record
)

create_and_save_forecast_plot(
    pred_df,
    algorithm_name,
    "Vanilla",
    "2021-09-26",
    "2021-12-31"
)

# GRU Model with bidirectionality
rnn_config = SingleStepRNNConfig(
    rnn_type="GRU",
    input_size=11,  # 25 for multivariate time series
    hidden_size=32,  # hidden size of the RNN
    num_layers=5, # number of layers
    bidirectional=True, # bidirectional RNN
    learning_rate=1e-3,
)
model = SingleStepRNNModel(rnn_config)
# model.float()

trainer = pl.Trainer(
    # logger=logger,
    min_epochs=5,
    max_epochs=100,
    accelerator = "gpu",
    devices = 1,
    callbacks=[EarlyStopping(monitor="valid_loss", patience=10)],
)
trainer.fit(model, datamodule)

algorithm_name = rnn_config.rnn_type

pred_df, metrics_vanilla = process_predictions_and_metrics(
    model, datamodule, algorithm_name, "Bidirectional", train, test, metric_record
)

create_and_save_forecast_plot(
    pred_df,
    algorithm_name,
    "Bidirectional",
    "2021-09-26",
    "2021-12-31"
)

# save the metrics to a CSV file so that we can compare the performance of different algorithms from different experiments
metrics_df = pd.DataFrame(metric_record)
metrics_df.to_csv(f"reports/metrics/{selected_area}_metrics.csv", index=False)
