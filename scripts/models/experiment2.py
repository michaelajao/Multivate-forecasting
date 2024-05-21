# Experiment 2 Script for COVID-19 Forecasting
# change path to the root directory of the project
import os
os.chdir("../../")

# Description: This script contains the code for the second experiment in the project, 
# forecasting COVID-19 MVBeds using various RNN models and hyperparameter tuning with Simulated Annealing.

# Imports for handling data
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import cycle

# Imports for machine learning
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse

# Imports for visualization
import plotly.express as px
import plotly.graph_objects as go

# Progress bar
from tqdm.autonotebook import tqdm
tqdm.pandas()

# Local imports for data loaders and models
from src.utils import plotting_utils
from src.dl.dataloaders import TimeSeriesDataModule
from src.dl.multivariate_models import SingleStepRNNConfig, SingleStepRNNModel, Seq2SeqConfig, Seq2SeqModel, RNNConfig
from src.transforms.target_transformations import AutoStationaryTransformer

# Set seeds for reproducibility
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
import logging

# Set logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Utility Functions

def format_plot(fig, legends=None, xlabel="Time", ylabel="Value", title="", font_size=15):
    """
    Formats the plot with given parameters.

    Parameters:
        fig (plotly.graph_objs._figure.Figure): The plotly figure object.
        legends (list): List of legends for the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        title (str): Title of the plot.
        font_size (int): Font size for labels and title.

    Returns:
        plotly.graph_objs._figure.Figure: Formatted plotly figure object.
    """
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
    """
    Calculates the Mean Absolute Scaled Error (MASE).

    Parameters:
        actual (array-like): Array of actual values.
        predicted (array-like): Array of predicted values.
        insample_actual (array-like): Array of in-sample actual values for scaling.

    Returns:
        float: MASE value.
    """
    mae_insample = np.mean(np.abs(np.diff(insample_actual)))
    mae_outsample = np.mean(np.abs(actual - predicted))
    return mae_outsample / mae_insample

def forecast_bias(actual, predicted):
    """
    Calculates the forecast bias.

    Parameters:
        actual (array-like): Array of actual values.
        predicted (array-like): Array of predicted values.

    Returns:
        float: Forecast bias value.
    """
    return np.mean(predicted - actual)

def plot_forecast(pred_df, forecast_columns, forecast_display_names=None, save_path=None):
    """
    Plots the forecast and actual values.

    Parameters:
        pred_df (pd.DataFrame): DataFrame containing the predictions and actual values.
        forecast_columns (list): List of column names containing forecast values.
        forecast_display_names (list): List of display names for the forecast columns.
        save_path (str): Path to save the plot. Default is None.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object with the forecast plot.
    """
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
    """
    Highlights the absolute minimum value in a Series.

    Parameters:
        s (pd.Series): Series of values.
        props (str): CSS properties for highlighting. Default is "".

    Returns:
        np.array: Array with highlighting properties.
    """
    return np.where(s == np.nanmin(np.abs(s.values)), props, "")

# Load and Prepare Data
data_path = Path("data/processed/merged_nhs_covid_data.csv")
data = pd.read_csv(data_path).drop("Unnamed: 0", axis=1)
data["date"] = pd.to_datetime(data["date"])

# Select and Process Data
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

# Add rolling features
def add_rolling_features(df, window_size, columns, agg_funcs=None):
    """
    Adds rolling window features to the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        window_size (int): Window size for the rolling calculation.
        columns (list): List of columns to apply the rolling calculation.
        agg_funcs (list): List of aggregation functions. Default is ["mean"].

    Returns:
        pd.DataFrame: DataFrame with added rolling features.
        dict: Dictionary of added rolling features.
    """
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

window_size = 7
columns_to_roll = ["hospitalCases", "newAdmissions", "new_confirmed", "new_deceased"]
agg_funcs = ["mean", "std"]

data_filtered, added_features = add_rolling_features(data_filtered, window_size, columns_to_roll, agg_funcs)

for column, features in added_features.items():
    logging.info(f"{column}: {', '.join(features)}")

# Add time-lagged features
def add_lags(data, lags, features):
    """
    Adds lagged features to the DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        lags (list): List of lag periods.
        features (list): List of features to apply the lag.

    Returns:
        pd.DataFrame: DataFrame with added lagged features.
        list: List of added lagged features.
    """
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

# Create temporal features
def create_temporal_features(df, date_column):
    """
    Creates temporal features from the date column.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        date_column (str): Name of the date column.

    Returns:
        pd.DataFrame: DataFrame with added temporal features.
    """
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["day_of_week"] = df[date_column].dt.dayofweek
    return df

data_filtered = create_temporal_features(data_filtered, "date")
data_filtered = data_filtered.set_index("date")

# Load SEIRD data
seird_data = pd.read_csv("reports/predictions.csv")
seird_data["date"] = pd.to_datetime(seird_data["date"])
seird_data = seird_data.set_index("date")

# Merge dataframes
merged_data = pd.merge(data_filtered, seird_data, left_index=True, right_index=True, how="inner")

# Set the target variable and make it stationary
target = "covidOccupiedMVBeds"
seasonal_period = 7
auto_stationary = AutoStationaryTransformer(seasonal_period=seasonal_period)
data_stat = auto_stationary.fit_transform(merged_data[[target]], freq="D")
merged_data[target] = data_stat.values

# Filter data between the specified dates
start_date = "2020-04-14"
end_date = "2020-12-30"
merged_data = merged_data[start_date:end_date]

min_date = merged_data.index.min()
max_date = merged_data.index.max()

# Calculate the range of dates
date_range = max_date - min_date
logging.info(f"Data ranges from {min_date} to {max_date} ({date_range.days} days)")

# Split the data into training, validation, and testing sets
train_end = min_date + pd.Timedelta(days=date_range.days * 0.45)
val_end = train_end + pd.Timedelta(days=date_range.days * 0.15)

train = merged_data[merged_data.index <= train_end]
val = merged_data[(merged_data.index > train_end) & (merged_data.index < val_end)]
test = merged_data[merged_data.index >= val_end]

# Concatenate the DataFrames
sample_df = pd.concat([train, val, test])

# Convert all the feature columns to float32
for col in sample_df.columns:
    sample_df[col] = sample_df[col].astype("float32")

columns_to_select = [
    "covidOccupiedMVBeds",
    "hospitalCases_rolling_7_mean",
    "hospitalCases_rolling_7_std",
    "newAdmissions_rolling_7_mean",
    "newAdmissions_rolling_7_std",
    "new_confirmed_rolling_7_mean",
    "new_confirmed_rolling_7_std",
    "new_deceased_rolling_7_mean",
    "new_deceased_rolling_7_std",
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

# Prepare DataModule for PyTorch Lightning
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

# Train Vanilla Model
rnn_config = SingleStepRNNConfig(
    rnn_type="RNN",
    input_size=len(columns_to_select),  # number of features
    hidden_size=32,  # hidden size of the RNN
    num_layers=5,  # number of layers
    bidirectional=False,  # bidirectional RNN
    learning_rate=1e-3,
)
model = SingleStepRNNModel(rnn_config)

trainer = pl.Trainer(
    min_epochs=5,
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    callbacks=[EarlyStopping(monitor="valid_loss", patience=10)],
)
trainer.fit(model, datamodule)

# Evaluate Vanilla Model
metric_record = []
predictions = trainer.predict(model, datamodule.test_dataloader())
predictions = torch.cat(predictions).squeeze().detach().numpy()
predictions = predictions * datamodule.train.std + datamodule.train.mean
actuals = test["covidOccupiedMVBeds"].values

assert actuals.shape == predictions.shape, "Mismatch in shapes between actuals and predictions"

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
logging.info(metrics)

# Simulated Annealing Hyperparameter Tuning

# Define the bounds for parameters
param_bounds = {
    "rnn_type": ["RNN", "GRU", "LSTM"],
    "hidden_size": (32, 128),  # Hidden size between 32 and 128
    "num_layers": (5, 30),
    "bidirectional": [True, False]
}

# Initial hyperparameters and temperature
initial_params = ["RNN", 32, 5, True]
initial_temp = 10

# Define the objective function
def objective(params):
    """
    Objective function for Simulated Annealing.

    Parameters:
        params (list): List of parameters [rnn_type, hidden_size, num_layers, bidirectional].

    Returns:
        float: Mean Absolute Error (MAE) of the model predictions.
    """
    rnn_type, hidden_size, num_layers, bidirectional = params
    rnn_config = SingleStepRNNConfig(
        rnn_type=rnn_type,
        input_size=len(columns_to_select),
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

    assert actuals.shape == predictions.shape, "Mismatch in shapes between actuals and predictions"

    return np.mean(np.abs(actuals - predictions))

def neighbor(params):
    """
    Generates a neighboring solution by perturbing the parameters.

    Parameters:
        params (list): Current list of parameters.

    Returns:
        list: New list of perturbed parameters.
    """
    rnn_type, hidden_size, num_layers, bidirectional = params

    hidden_size = np.random.randint(*param_bounds["hidden_size"])
    num_layers = np.random.randint(*param_bounds["num_layers"])
    rnn_type = np.random.choice(param_bounds["rnn_type"])
    bidirectional = bool(np.random.choice(param_bounds["bidirectional"]))

    return [rnn_type, hidden_size, num_layers, bidirectional]

def simulated_annealing(objective, initial_params, initial_temp, neighbor, n_iter, cooling_rate=0.20, verbose=True):
    """
    Simulated Annealing optimization algorithm.

    Parameters:
        objective (function): The objective function to minimize.
        initial_params (list): Initial list of parameters.
        initial_temp (float): Initial temperature.
        neighbor (function): Function to generate neighboring solutions.
        n_iter (int): Number of iterations.
        cooling_rate (float): Cooling rate for the temperature.
        verbose (bool): If True, print iteration details.

    Returns:
        tuple: Best cost, best parameters, and cost history.
    """
    current_params = initial_params
    current_cost = objective(current_params)
    best_params = current_params
    best_cost = current_cost
    temp = initial_temp
    cost_history = []

    for i in range(n_iter):
        candidate_params = neighbor(current_params)
        candidate_cost = objective(candidate_params)

        acceptance_probability = np.exp(-abs(candidate_cost - current_cost) / temp)

        if candidate_cost < current_cost or np.random.uniform() < acceptance_probability:
            current_params = candidate_params
            current_cost = candidate_cost

            if current_cost < best_cost:
                best_params = current_params
                best_cost = current_cost

        temp *= cooling_rate
        cost_history.append(best_cost)

        logging.info(f"Iteration: {i+1}, Best Cost: {best_cost:.4f}, Current Cost: {current_cost:.4f}, Temperature: {temp:.4f}")

        if i > 10 and np.all(np.isclose(cost_history[-10:], cost_history[-1])):
            logging.info(f"Early stopping at iteration {i+1} due to convergence.")
            break

    return best_cost, best_params, cost_history

# Run Simulated Annealing for 100 iterations
initial_params = ["RNN", 32, 5, True]
initial_temp = 10
n_iter = 100
cooling_rate = 0.95

best_cost, best_params, cost_history = simulated_annealing(
    objective, initial_params, initial_temp, neighbor, n_iter, cooling_rate
)

logging.info(f"Best Parameters: {best_params}, Best Cost: {best_cost}")

# Plot the MAEs gotten from the SA
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1, len(cost_history) + 1), y=cost_history, mode="lines"))
fig.update_layout(
    title="Simulated Annealing Optimization for Hyperparameter Tuning",
    xaxis_title="Iteration",
    yaxis_title="Best Cost",
    template="plotly_white",
)

save_path = f"reports/figures/{selected_area}_sa_optimization.pdf"
pio.write_image(fig, save_path)
fig.show()

# Prediction using the best parameters
rnn_config = SingleStepRNNConfig(
    rnn_type=best_params[0],
    input_size=len(columns_to_select),
    hidden_size=best_params[1],
    num_layers=best_params[2],
    bidirectional=best_params[3],
    learning_rate=1e-3,
)

model = SingleStepRNNModel(rnn_config)

trainer = pl.Trainer(
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

assert actuals.shape == predictions.shape, "Mismatch in shapes between actuals and predictions"

algorithm_name = rnn_config.rnn_type

metrics = {
    "Algorithm": f"{algorithm_name} (SA)",
    "MAE": mae(actuals, predictions),
    "MSE": mse(actuals, predictions),
    "MASE": mase(actuals, predictions, train["covidOccupiedMVBeds"].values),
    "Forecast Bias": forecast_bias(actuals, predictions),
}

metrics = {
    key: format_.format(value)
    for key, value, format_ in zip(metrics.keys(), metrics.values(), value_formats)
}

pred_df_ = pd.DataFrame({f"Optimized {algorithm_name} (SA)": predictions}, index=test.index)
pred_df = test.join(pred_df_)
metric_record.append(metrics)

logging.info(metrics)

# Seq2Seq Model Training and Evaluation 

HORIZON = 1
WINDOW = 7

encoder_config = RNNConfig(
    input_size=len(columns_to_select),
    hidden_size=32,
    num_layers=5,
    bidirectional=True
)

decoder_config = RNNConfig(
    input_size=len(columns_to_select),
    hidden_size=32,
    num_layers=5,
    bidirectional=True
)

rnn2fc_config = Seq2SeqConfig(
    encoder_type="LSTM",
    decoder_type="FC",
    encoder_params=encoder_config,
    decoder_params={"window_size": WINDOW, "horizon": HORIZON},
    decoder_use_all_hidden=True,
    learning_rate=1e-3,
)

model = Seq2SeqModel(rnn2fc_config)

trainer = pl.Trainer(
    min_epochs=5,
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    callbacks=[EarlyStopping(monitor="valid_loss", patience=10)],
)

trainer.fit(model, datamodule)

# Evaluate Seq2Seq Model
tag = f"{rnn2fc_config.encoder_type}_{rnn2fc_config.decoder_type}_{'all_hidden' if rnn2fc_config.decoder_use_all_hidden else 'last_hidden'}"

predictions = trainer.predict(model, datamodule.test_dataloader())
predictions = torch.cat(predictions).squeeze().detach().numpy()
predictions = predictions * datamodule.train.std + datamodule.train.mean

actuals = test["covidOccupiedMVBeds"].values

assert actuals.shape == predictions.shape, "Mismatch in shapes between actuals and predictions"

algorithm_name = rnn2fc_config.encoder_type

metrics = {
    "Algorithm": f"Seq2Seq {algorithm_name}",
    "MAE": mae(actuals, predictions),
    "MSE": mse(actuals, predictions),
    "MASE": mase(actuals, predictions, train["covidOccupiedMVBeds"].values),
    "Forecast Bias": forecast_bias(actuals, predictions),
}

metrics = {
    key: format_.format(value)
    for key, value, format_ in zip(metrics.keys(), metrics.values(), value_formats)
}

pred_df_ = pd.DataFrame({f"Seq2Seq {algorithm_name}": predictions}, index=test.index)
pred_df = test.join(pred_df_)

metric_record.append(metrics)
logging.info(metrics)

# Simulated Annealing for Seq2Seq model hyperparameter tuning
param_bounds = {
    "encoder_type": ["RNN", "GRU", "LSTM"],
    "decoder_type": ["FC"],
    "hidden_size": (32, 128),
    "num_layers": (5, 30),
    "bidirectional": [True, False],
    "decoder_use_all_hidden": [True, False],
}

initial_params = ["RNN", "FC", 32, 5, True, True]

initial_temp = 10
n_iter = 100
cooling_rate = 0.95

def objective(params):
    """
    Objective function for Seq2Seq Simulated Annealing.

    Parameters:
        params (list): List of parameters [encoder_type, decoder_type, hidden_size, num_layers, bidirectional, decoder_use_all_hidden].

    Returns:
        float: Mean Absolute Error (MAE) of the model predictions.
    """
    encoder_type, decoder_type, hidden_size, num_layers, bidirectional, decoder_use_all_hidden = params

    encoder_config = RNNConfig(
        input_size=len(columns_to_select),
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional
    )

    decoder_config = RNNConfig(
        input_size=len(columns_to_select),
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional
    )

    rnn2fc_config = Seq2SeqConfig(
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        encoder_params=encoder_config,
        decoder_params={"window_size": WINDOW, "horizon": HORIZON},
        decoder_use_all_hidden=decoder_use_all_hidden,
        learning_rate=1e-3,
    )

    model = Seq2SeqModel(rnn2fc_config)

    trainer = pl.Trainer(
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

    assert actuals.shape == predictions.shape, "Mismatch in shapes between actuals and predictions"

    return np.mean(np.abs(actuals - predictions))

def neighbor(params):
    """
    Generates a neighboring solution by perturbing the parameters for Seq2Seq.

    Parameters:
        params (list): Current list of parameters.

    Returns:
        list: New list of perturbed parameters.
    """
    encoder_type, decoder_type, hidden_size, num_layers, bidirectional, decoder_use_all_hidden = params

    hidden_size = np.random.randint(*param_bounds["hidden_size"])
    num_layers = np.random.randint(*param_bounds["num_layers"])
    encoder_type = np.random.choice(param_bounds["encoder_type"])
    decoder_type = np.random.choice(param_bounds["decoder_type"])
    bidirectional = bool(np.random.choice(param_bounds["bidirectional"]))
    decoder_use_all_hidden = bool(np.random.choice(param_bounds["decoder_use_all_hidden"]))

    return [encoder_type, decoder_type, hidden_size, num_layers, bidirectional, decoder_use_all_hidden]

best_cost, best_params, cost_history = simulated_annealing(
    objective, initial_params, initial_temp, neighbor, n_iter, cooling_rate
)

logging.info(f"Best Parameters: {best_params}, Best Cost: {best_cost}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1, len(cost_history) + 1), y=cost_history, mode="lines"))
fig.update_layout(
    title="Simulated Annealing Optimization Seq2Seq for Hyperparameter Tuning",
    xaxis_title="Iteration",
    yaxis_title="Best Cost",
    template="plotly_white",
)

save_path = f"reports/figures/{selected_area}_sa_seq2seq_optimization.pdf"
pio.write_image(fig, save_path)
fig.show()

# Prediction using the best parameters
encoder_type, decoder_type, hidden_size, num_layers, bidirectional, decoder_use_all_hidden = best_params

encoder_config = RNNConfig(
    input_size=len(columns_to_select),
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=bidirectional
)

decoder_config = RNNConfig(
    input_size=len(columns_to_select),
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=bidirectional
)

rnn2fc_config = Seq2SeqConfig(
    encoder_type=encoder_type,
    decoder_type=decoder_type,
    encoder_params=encoder_config,
    decoder_params={"window_size": WINDOW, "horizon": HORIZON},
    decoder_use_all_hidden=decoder_use_all_hidden,
    learning_rate=1e-3,
)

model = Seq2SeqModel(rnn2fc_config)

trainer = pl.Trainer(
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

assert actuals.shape == predictions.shape, "Mismatch in shapes between actuals and predictions"

algorithm_name = encoder_type

metrics = {
    "Algorithm": f"Seq2Seq {algorithm_name} (SA)",
    "MAE": mae(actuals, predictions),
    "MSE": mse(actuals, predictions),
    "MASE": mase(actuals, predictions, train["covidOccupiedMVBeds"].values),
    "Forecast Bias": forecast_bias(actuals, predictions),
}

metrics = {
    key: format_.format(value)
    for key, value, format_ in zip(metrics.keys(), metrics.values(), value_formats)
}

pred_df_ = pd.DataFrame({f"Optimized Seq2Seq {algorithm_name} (SA)": predictions}, index=test.index)
pred_df = test.join(pred_df_)
metric_record.append(metrics)

logging.info(metrics)

shutil.rmtree("lightning_logs")

# Save final metrics
metric_df = pd.DataFrame(metric_record)
metric_df[["MAE", "MSE", "MASE", "Forecast Bias"]] = metric_df[
    ["MAE", "MSE", "MASE", "Forecast Bias"]
].astype("float32")

final_metric_file = f"reports/results/{selected_area}_final_metrics.csv"
metric_df.to_csv(final_metric_file, index=False)

# Plot all the forecasts together vs actuals for comparison
def plot_all_forecasts(pred_df, actual_col, forecast_columns, forecast_display_names=None, save_path=None):
    """
    Plots all forecasts together vs actuals for comparison.

    Parameters:
        pred_df (pd.DataFrame): DataFrame containing the predictions and actual values.
        actual_col (str): Column name of the actual values.
        forecast_columns (list): List of column names containing forecast values.
        forecast_display_names (list): List of display names for the forecast columns.
        save_path (str): Path to save the plot. Default is None.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object with the comparison plot.
    """
    if forecast_display_names is None:
        forecast_display_names = forecast_columns
    else:
        assert len(forecast_columns) == len(forecast_display_names)

    mask = ~pred_df[forecast_columns[0]].isnull()
    colors = px.colors.qualitative.Set2
    act_color = colors[0]
    colors = cycle(colors[1:])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pred_df[mask].index,
            y=pred_df[mask][actual_col],
            mode="lines",
            marker=dict(size=6, opacity=0.5),
            line=dict(color=act_color, width=2),
            name="Actual COVID-19 MVBeds trends",
        )
    )

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

    fig.update_layout(
        title="Comparison of COVID-19 MVBeds Forecasts vs Actuals",
        xaxis_title="Date",
        yaxis_title="COVID-19 MVBeds",
        template="plotly_white",
    )

    if save_path:
        pio.write_image(fig, save_path)
        logging.info(f"Plot saved to {save_path}")
    
    fig.show()

# Example usage
forecast_columns = [
    f"Vanilla {algorithm_name}", 
    f"Optimized {algorithm_name} (SA)", 
    f"Seq2Seq {algorithm_name}", 
    f"Optimized Seq2Seq {algorithm_name} (SA)"
]

forecast_display_names = [
    f"Vanilla {algorithm_name}", 
    f"Optimized {algorithm_name} (SA)", 
    f"Seq2Seq {algorithm_name}", 
    f"Optimized Seq2Seq {algorithm_name} (SA)"
]

plot_all_forecasts(
    pred_df, 
    actual_col="covidOccupiedMVBeds", 
    forecast_columns=forecast_columns, 
    forecast_display_names=forecast_display_names, 
    save_path=f"reports/figures/{selected_area}_forecast_multivariate_comparison.pdf"
)