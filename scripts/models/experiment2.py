import os
os.chdir("../../")

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

import plotly.express as px
import plotly.graph_objects as go

from tqdm.autonotebook import tqdm
tqdm.pandas()

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

import plotly.io as pio
pio.templates.default = "plotly_white"

import warnings
warnings.filterwarnings("ignore")
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def format_plot(fig, legends=None, xlabel="Time", ylabel="Value", title="", font_size=15):
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

def plot_forecast(pred_df, forecast_columns, forecast_display_names=None, save_path=None):
    if forecast_display_names is None:
        forecast_display_names = forecast_columns
    else:
        assert len(forecast_columns) == len(forecast_display_names)

    mask = ~pred_df[forecast_columns[0]].isnull()
    colors = px.colors.qualitative.Set2  # Using a different color palette
    act_color = colors[0]
    colors = cycle(colors[1:])

    fig = go.Figure()

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

    fig = format_plot(fig, xlabel="Date", ylabel="COVID-19 MVBeds", title="COVID-19 MVBeds Forecast Comparison")

    if save_path:
        pio.write_image(fig, save_path)
    return fig

def highlight_abs_min(s, props=""):
    return np.where(s == np.nanmin(np.abs(s.values)), props, "")

data_path = Path("data/processed/merged_nhs_covid_data.csv")
data = pd.read_csv(data_path).drop("Unnamed: 0", axis=1)
data["date"] = pd.to_datetime(data["date"])

selected_area = "South West"
data_filtered = data[data["areaName"] == selected_area]

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

window_size = 7
columns_to_roll = ["hospitalCases", "newAdmissions", "new_confirmed", "new_deceased"]
agg_funcs = ["mean", "std"]

data_filtered, added_features = add_rolling_features(data_filtered, window_size, columns_to_roll, agg_funcs)

for column, features in added_features.items():
    logging.info(f"{column}: {', '.join(features)}")

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
data_filtered = data_filtered.set_index("date")

seird_data = pd.read_csv("reports/predictions.csv")
seird_data["date"] = pd.to_datetime(seird_data["date"])
seird_data = seird_data.set_index("date")

merged_data = pd.merge(data_filtered, seird_data, left_index=True, right_index=True, how="inner")

target = "covidOccupiedMVBeds"
seasonal_period = 7
auto_stationary = AutoStationaryTransformer(seasonal_period=seasonal_period)
data_stat = auto_stationary.fit_transform(merged_data[[target]], freq="D")
merged_data[target] = data_stat.values

start_date = "2020-04-14"
end_date = "2020-12-30"
merged_data = merged_data[start_date:end_date]

min_date = merged_data.index.min()
max_date = merged_data.index.max()

date_range = max_date - min_date
logging.info(f"Data ranges from {min_date} to {max_date} ({date_range.days} days)")

train_end = min_date + pd.Timedelta(days=date_range.days * 0.45)
val_end = train_end + pd.Timedelta(days=date_range.days * 0.15)

train = merged_data[merged_data.index <= train_end]
val = merged_data[(merged_data.index > train_end) & (merged_data.index < val_end)]
test = merged_data[merged_data.index >= val_end]

sample_df = pd.concat([train, val, test])

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

datamodule = TimeSeriesDataModule(
    data=sample_df,
    n_val=val.shape[0],
    n_test=test.shape[0],
    window=7,
    horizon=1,
    normalize="global",
    batch_size=32,
    num_workers=0,
)
datamodule.setup()

pred_df = test.copy()

def train_and_evaluate(model, rnn_config, datamodule, test, train, algorithm_name, metric_record, pred_df):
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

    pred_df_ = pd.DataFrame({algorithm_name: predictions}, index=test.index)
    pred_df = pred_df.join(pred_df_)
    metric_record.append(metrics)
    logging.info(metrics)
    return metric_record, pred_df

rnn_config = SingleStepRNNConfig(
    rnn_type="RNN",
    input_size=len(columns_to_select),
    hidden_size=32,
    num_layers=5,
    bidirectional=False,
    learning_rate=1e-3,
)
model = SingleStepRNNModel(rnn_config)
algorithm_name = "Vanilla RNN"
metric_record, pred_df = train_and_evaluate(model, rnn_config, datamodule, test, train, algorithm_name, metric_record=[], pred_df=pred_df)

rnn_config = SingleStepRNNConfig(
    rnn_type="LSTM",
    input_size=len(columns_to_select),
    hidden_size=32,
    num_layers=5,
    bidirectional=False,
    learning_rate=1e-3,
)
model = SingleStepRNNModel(rnn_config)
algorithm_name = "Vanilla LSTM"
metric_record, pred_df = train_and_evaluate(model, rnn_config, datamodule, test, train, algorithm_name, metric_record, pred_df)

rnn_config = SingleStepRNNConfig(
    rnn_type="GRU",
    input_size=len(columns_to_select),
    hidden_size=32,
    num_layers=5,
    bidirectional=False,
    learning_rate=1e-3,
)
model = SingleStepRNNModel(rnn_config)
algorithm_name = "Vanilla GRU"
metric_record, pred_df = train_and_evaluate(model, rnn_config, datamodule, test, train, algorithm_name, metric_record, pred_df)

param_bounds = {
    "rnn_type": ["RNN", "GRU", "LSTM"],
    "hidden_size": (32, 128),
    "num_layers": (5, 30),
    "bidirectional": [True, False]
}

initial_params = ["RNN", 32, 5, True]
initial_temp = 10

def objective(params):
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
    if len(params) == 4:
        rnn_type, hidden_size, num_layers, bidirectional = params
        hidden_size = np.random.randint(*param_bounds["hidden_size"])
        num_layers = np.random.randint(*param_bounds["num_layers"])
        rnn_type = np.random.choice(param_bounds["rnn_type"])
        bidirectional = bool(np.random.choice(param_bounds["bidirectional"]))
        return [rnn_type, hidden_size, num_layers, bidirectional]
    elif len(params) == 6:
        encoder_type, decoder_type, hidden_size, num_layers, bidirectional, decoder_use_all_hidden = params
        hidden_size = np.random.randint(*param_bounds["hidden_size"])
        num_layers = np.random.randint(*param_bounds["num_layers"])
        encoder_type = np.random.choice(param_bounds["encoder_type"])
        decoder_type = np.random.choice(param_bounds["decoder_type"])
        bidirectional = bool(np.random.choice(param_bounds["bidirectional"]))
        decoder_use_all_hidden = bool(np.random.choice(param_bounds["decoder_use_all_hidden"]))
        return [encoder_type, decoder_type, hidden_size, num_layers, bidirectional, decoder_use_all_hidden]

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

    return best_cost, best_params, cost_history

n_iter = 100
cooling_rate = 0.95

best_cost, best_params, cost_history = simulated_annealing(
    objective, initial_params, initial_temp, neighbor, n_iter, cooling_rate
)

logging.info(f"Best Parameters: {best_params}, Best Cost: {best_cost}")

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

rnn_config = SingleStepRNNConfig(
    rnn_type=best_params[0],
    input_size=len(columns_to_select),
    hidden_size=best_params[1],
    num_layers=best_params[2],
    bidirectional=best_params[3],
    learning_rate=1e-3,
)

model = SingleStepRNNModel(rnn_config)
algorithm_name = f"Optimized {rnn_config.rnn_type} (SA)"
metric_record, pred_df = train_and_evaluate(model, rnn_config, datamodule, test, train, algorithm_name, metric_record, pred_df)

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
algorithm_name = "Seq2Seq LSTM"
metric_record, pred_df = train_and_evaluate(model, rnn2fc_config, datamodule, test, train, algorithm_name, metric_record, pred_df)

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

def objective(params):
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
algorithm_name = f"Optimized Seq2Seq {encoder_type} (SA)"
metric_record, pred_df = train_and_evaluate(model, rnn2fc_config, datamodule, test, train, algorithm_name, metric_record, pred_df)

shutil.rmtree("lightning_logs")

metric_df = pd.DataFrame(metric_record)
metric_df[["MAE", "MSE", "MASE", "Forecast Bias"]] = metric_df[
    ["MAE", "MSE", "MASE", "Forecast Bias"]
].astype("float32")

final_metric_file = f"reports/results/{selected_area}_final_metrics.csv"
metric_df.to_csv(final_metric_file, index=False)

forecast_columns = [
    "Vanilla RNN", 
    "Vanilla LSTM", 
    "Vanilla GRU", 
    "Optimized RNN (SA)", 
    "Seq2Seq LSTM", 
    "Optimized Seq2Seq LSTM (SA)"
]

forecast_display_names = [
    "Vanilla RNN", 
    "Vanilla LSTM", 
    "Vanilla GRU", 
    "Optimized RNN (SA)", 
    "Seq2Seq LSTM", 
    "Optimized Seq2Seq LSTM (SA)"
]

comparison_fig = plot_forecast(pred_df, forecast_columns, forecast_display_names, save_path="reports/figures/forecast_comparison.pdf")
comparison_fig.show()
