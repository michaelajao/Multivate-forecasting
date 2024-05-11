# %%
# Path to source code
%cd ../../

# %%
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
from src.dl.multivariate_models import SingleStepRNNConfig, SingleStepRNNModel, Seq2SeqConfig, Seq2SeqModel, RNNConfig
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

# %%
# check the unique values in the areaName column
data["areaName"].unique()

# %%
# Select a different area name
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

# %%
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
    # Drop rows with NaN values which are the result of rolling window
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

# Print added features for each column
for column, features in added_features.items():
    print(f"{column}: {', '.join(features)}")

# %%
# Define a function to add time-lagged features to the dataset
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
data_filtered.head()

# %%
data_filtered["date"] = pd.to_datetime(data_filtered["date"])
data_filtered = data_filtered.set_index("date")
data_filtered.head()

# %%
# Set the target variable
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

# %%
train_dates = (train.index.min(), train.index.max())
val_dates = (val.index.min(), val.index.max())
test_dates = (test.index.min(), test.index.max())

print(f"Train dates: {train_dates}, Val dates: {val_dates}, Test dates: {test_dates}")

# %%
# plot the train data
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=train.index, y=train["covidOccupiedMVBeds"], mode="lines", name="Train Data"
    )
)
fig.update_layout(
    title="Train Data for COVID-19 Occupied MV Beds in Midlands Region",
    xaxis_title="Date",
    yaxis_title="COVID-19 Occupied MV Beds",
    template="plotly_white",
)
fig.show()

# %%
# Concatenate the DataFrames
sample_df = pd.concat([train, val, test])

# Convert all the feature columns to float32
for col in sample_df.columns:
    sample_df[col] = sample_df[col].astype("float32")

sample_df.info()

# %%
sample_df.columns

# %%
columns_to_select = [
    "covidOccupiedMVBeds",
    # "hospitalCases",
    # "newAdmissions",
    # "new_confirmed",
    # "new_deceased",
    # "hospitalCases_rolling_7_mean",
    # "hospitalCases_rolling_7_std",
    # "newAdmissions_rolling_7_mean",
    # "newAdmissions_rolling_7_std",
    # "new_confirmed_rolling_7_mean",
    # "new_confirmed_rolling_7_std",
    # "new_deceased_rolling_7_mean",
    # "new_deceased_rolling_7_std",
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

# %%
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

# %%
# Setting up TensorBoard logger
# logger = TensorBoardLogger("notebook/multivariate/tb_logs", name="my_rnn_experiment")

# %tensorboard --logdir tb_logs


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

# %%
metric_record = []

# %%
predictions = trainer.predict(model, datamodule.test_dataloader())
predictions = torch.cat(predictions).squeeze().detach().numpy()
# De-normalizing the predictions
predictions = predictions * datamodule.train.std + datamodule.train.mean

actuals = test["covidOccupiedMVBeds"].values

assert (
    actuals.shape == predictions.shape
), "Mismatch in shapes between actuals and predictions"

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

pred_df_ = pd.DataFrame({f"Vanilla {algorithm_name}": predictions}, index=test.index)
pred_df = test.join(pred_df_)

metric_record.append(metrics)
print(metrics)

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
    save_path = f"reports/images/forecast_multivariate_{experiment_type}_{algorithm_name}.pdf"
    pio.write_image(fig, save_path)
    fig.show()
    
create_and_save_forecast_plot(pred_df, algorithm_name, "Vanilla", "2021-09-26", "2021-12-31")

# %%
# save model to reuse later, in the report folder 
model_path = Path("models")
model_path.mkdir(exist_ok=True)
model_file = model_path / f"{algorithm_name}_model.pt"
torch.save(model.state_dict(), model_file)

# save the metrics to a csv file
metric_df = pd.DataFrame(metric_record)
metric_file = model_path / f"{algorithm_name}_metrics.csv"

metric_df.to_csv(metric_file, index=False)

# %% [markdown]
# ## Simulated annealing hyperparameter tunning

# %%
# Define the bounds for parameters
param_bounds = {
    "rnn_type": ["RNN", "GRU", "LSTM"],
    "hidden_size": (32, 128),  # Hidden size between 32 and 128
    "num_layers": (5, 30),
    "bidirectional": [True, False]
}

# Initial hyperparameters and temperature
initial_params = ["RNN", 64, 10, True]  # Updated for a realistic hidden size initialization
initial_temp = 10

# %%
# Define the objective function
def objective(params):
    rnn_type, hidden_size, num_layers, bidirectional = params
    rnn_config = SingleStepRNNConfig(
        rnn_type=rnn_type,
        input_size=11,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        learning_rate=1e-3
    )
    model = SingleStepRNNModel(rnn_config)
    model.float()

    trainer = pl.Trainer(
        min_epochs=5,
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="valid_loss", patience=10)],
    )
    trainer.fit(model, datamodule)
    
    shutil.rmtree("lightning_logs")

    predictions = trainer.predict(model, datamodule.test_dataloader())
    predictions = torch.cat(predictions).squeeze().detach().numpy()
    predictions = predictions * datamodule.train.std + datamodule.train.mean

    actuals = test["covidOccupiedMVBeds"].values

    assert (
        actuals.shape == predictions.shape
    ), "Mismatch in shapes between actuals and predictions"

    return np.mean(np.abs(actuals - predictions))  # Return the MAE


# %%
def neighbor(params):
    rnn_type, hidden_size, num_layers, bidirectional = params

    # Perturbations
    hidden_size = np.random.randint(*param_bounds["hidden_size"])
    num_layers = np.random.randint(*param_bounds["num_layers"])
    # learning_rate = np.random.uniform(*param_bounds["learning_rate"])
    rnn_type = np.random.choice(param_bounds["rnn_type"])
    bidirectional = bool(
        np.random.choice(param_bounds["bidirectional"])
    )  # Convert to native boolean
    
    # ensure that it returns the best parameters
    return [rnn_type, hidden_size, num_layers, bidirectional]

# Simulated Annealing Algorithm, measuring cost and temperature
def simulated_annealing(objective, initial_params, initial_temp, neighbor, n_iter, cooling_rate=0.20, verbose=True):
    current_params = initial_params
    current_cost = objective(current_params)
    best_params = current_params
    best_cost = current_cost
    temp = initial_temp
    cost_history = []

    for i in range(n_iter):
        candidate_params = neighbor(current_params)
        candidate_cost = objective(candidate_params)

        # Calculate the probability of accepting the new solution
        acceptance_probability = np.exp(-abs(candidate_cost - current_cost) / temp)

        # Decision to accept the new candidate
        if candidate_cost < current_cost or np.random.uniform() < acceptance_probability:
            current_params = candidate_params
            current_cost = candidate_cost

            # Update the best found solution
            if current_cost < best_cost:
                best_params = current_params
                best_cost = current_cost

        # Cooling down the temperature
        temp *= cooling_rate
        cost_history.append(best_cost)

        # Output current iteration details
        if verbose:
            print(f"Iteration: {i+1}, Best Cost: {best_cost:.4f}, Current Cost: {current_cost:.4f}, Temperature: {temp:.4f}")

        # Break early if the minimum cost has been constant for 5 iterations
        if i >= 5 and all(x == best_cost for x in cost_history[-5:]):
            print("Early stopping as there is no improvement in the last 5 iterations.")
            break

    return best_cost, best_params, cost_history

# %%
# Run Simulated Annealing for 100 iterations
initial_params = ["RNN", 64, 10, True]  # Initial parameters
initial_temp = 10
n_iter = 100
cooling_rate = 0.95  # More gradual cooling

best_cost, best_params, cost_history = simulated_annealing(
    objective, initial_params, initial_temp, neighbor, n_iter, cooling_rate
)

# %%
# Print the best parameters and best cost
print(f"Best Parameters: {best_params}, Best Cost: {best_cost}")

# %%
# plot the MAEs gotten from the SA
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1, len(cost_history) + 1), y=cost_history, mode="lines"))
fig.update_layout(
    title="Simulated Annealing Optimization for Hyperparameter Tuning",
    xaxis_title="Iteration",
    yaxis_title="Best Cost",
    template="plotly_white",
)
fig.show()

# %%
# prediction using the best parameters
rnn_config = SingleStepRNNConfig(
    rnn_type=best_params[0],
    input_size=11,
    hidden_size=best_params[1],
    num_layers=best_params[2],
    bidirectional=best_params[3],
    learning_rate=1e-3,
)

model = SingleStepRNNModel(rnn_config)
# model.float()

# logger = TensorBoardLogger("notebook/multivariate/tb_logs", name="my_rnn_experiment")
trainer = pl.Trainer(
    # logger=logger,
    min_epochs=5,
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    callbacks=[EarlyStopping(monitor="valid_loss", patience=10)],
)

trainer.fit(model, datamodule)

predictions = trainer.predict(model, datamodule.test_dataloader())
predictions = torch.cat(predictions).squeeze().detach().numpy()
predictions = predictions * datamodule.train.std + datamodule.train.mean

actuals = test["covidOccupiedMVBeds"].values

assert (
    actuals.shape == predictions.shape
), "Mismatch in shapes between actuals and predictions"

algorithm_name = rnn_config.rnn_type

metrics = {
    "Algorithm": f"{algorithm_name} (SA)",
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

pred_df_ = pd.DataFrame({f"Optimized {algorithm_name} (SA)": predictions}, index=test.index)

pred_df = test.join(pred_df_)

metric_record.append(metrics)

print(metrics)

fig = plot_forecast(
    pred_df,
    forecast_columns=[f"Optimized {algorithm_name} (SA)"],
    forecast_display_names=[f"Optimized {algorithm_name} (SA)"],
)

title = f"Forecasting COVID-19 MVBeds with {algorithm_name} (SA)"
fig = format_plot(fig, title=title)
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list(
            [
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all"),
            ]
        )
    ),
)

# save_path = f"images/forecast_multivarate_{algorithm_name}.png"
# pio.write_image(fig, save_path)
fig.show()

# %%
# shutil.rmtree("notebook/multivariate/tb_logs")

# %%
# Plotting the forecast
fig = plot_forecast(
    # simulated annealing model
    pred_df,
    forecast_columns=[f"Optimized {algorithm_name} (SA)"],
    forecast_display_names=[f"Optimized {algorithm_name} (SA)"],
    # forecast_columns=[f"{algorithm_name} (SA)"],
    # forecast_display_names=[algorithm_name],
)

title = f"Forecasting multivarate COVID-19 MVBeds with {algorithm_name} (SA)"
fig = format_plot(fig, title=title)

fig.update_xaxes(
    type="date", range=["2021-09-26", "2021-12-31"], dtick="M1", tickformat="%b %Y"
)

save_path = f"images/forecast_multivarate_{algorithm_name}_sa.png"
# pio.write_image(fig, save_path)
fig.show()

# %%


# %%


# %%


# %%
# save model to reuse later, in the report folder
model_file = model_path / f"{algorithm_name}_sa_model.pt"
torch.save(model.state_dict(), model_file)

# save the metrics to a csv file
metric_file = model_path / f"{algorithm_name}_sa_metrics.csv"
metric_df.to_csv(metric_file, index=False)

# %%
HORIZON = 1
WINDOW = 7

# %%
encoder_config = RNNConfig(
    input_size=11,  # Replace with actual number of input features
    hidden_size=32,  # Example size
    num_layers=5,
    bidirectional=True
)

decoder_config = RNNConfig(
    input_size=11,  # Should align with the encoder's output dimension
    hidden_size=32,  # Example size
    num_layers=5,
    bidirectional=True
)

rnn2fc_config = Seq2SeqConfig(
    encoder_type="GRU",
    decoder_type="FC",
    encoder_params=encoder_config,
    decoder_params={"window_size": WINDOW, "horizon": HORIZON},
    decoder_use_all_hidden=True,
    learning_rate=1e-3,
)

model = Seq2SeqModel(rnn2fc_config)

trainer = pl.Trainer(
    # logger=logger,
    min_epochs=5,
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    callbacks=[EarlyStopping(monitor="valid_loss", patience=10)],
)

trainer.fit(model, datamodule)

# %%
tag = f"{rnn2fc_config.encoder_type}_{rnn2fc_config.decoder_type}_{'all_hidden' if rnn2fc_config.decoder_use_all_hidden else 'last_hidden'}"

predictions = trainer.predict(model, datamodule.test_dataloader())
predictions = torch.cat(predictions).squeeze().detach().numpy()

predictions = predictions * datamodule.train.std + datamodule.train.mean

actuals = test["covidOccupiedMVBeds"].values

assert (
    actuals.shape == predictions.shape
), "Mismatch in shapes between actuals and predictions"

algorithm_name = rnn2fc_config.encoder_type

metrics = {
    "Algorithm": f"Seq2Seq {algorithm_name}",
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

pred_df_ = pd.DataFrame({f"Seq2Seq {algorithm_name}": predictions}, index=test.index)

pred_df = test.join(pred_df_)

metric_record.append(metrics)

print(metrics)

# %%
fig = plot_forecast(
    pred_df,
    forecast_columns=[f"Seq2Seq {algorithm_name}"],
    forecast_display_names=[f"Seq2Seq {algorithm_name}"],
)

title = f"Forecasting COVID-19 MVBeds with Seq2Seq {algorithm_name}"
fig = format_plot(fig, title=title)
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list(
            [
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all"),
            ]
        )
    ),
)

# save_path = f"images/forecast_multivarate_{algorithm_name}.png"
# pio.write_image(fig, save_path)
fig.show()

# %%
metric_df = pd.DataFrame(metric_record)
metric_df.info()

# %%
metric_df[["MAE", "MSE", "MASE", "Forecast Bias"]] = metric_df[
    ["MAE", "MSE", "MASE", "Forecast Bias"]
].astype("float32")

# %%
formatted = metric_df.style.format(
    {
        "MAE": "{:.4f}",
        "MSE": "{:.4f}",
        "MASE": "{:.4f}",
        "Forecast Bias": "{:.2f}%",
        "Time Elapsed": "{:.6f}",
    }
)
formatted = formatted.highlight_min(
    color="lightgreen", subset=["MAE", "MSE", "MASE"]
).apply(
    highlight_abs_min,
    props="color:black;background-color:lightgreen",
    axis=0,
    subset=["Forecast Bias"],
)
formatted

# %%



