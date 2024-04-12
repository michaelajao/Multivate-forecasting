%cd ../..

import os
import shutil
import numpy as np
import pandas as pd

# import plottnig libraries
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go

from pathlib import Path

#
from itertools import cycle
# import libraries from src
from src.utils import plotting_utils, ts_utils
from darts import TimeSeries
import darts
from darts.models import (
    NaiveSeasonal,
    NaiveMean,
    NaiveDrift,
    ExponentialSmoothing,
    AutoARIMA,
    ARIMA,
    Theta,
    FFT
)

from darts.metrics import mase, mse, mae, ope
from tqdm import tqdm

from src.utils.ts_utils import forecast_bias
from src.utils.general import LogTime
from src.utils import plotting_utils

np.random.seed(42)
tqdm.pandas()

os.makedirs('reports/results', exist_ok=True)

# Load the data 
preprocessed = Path('data/processed')
output = Path('reports/results')  

# from collections import namedtuple

# FeatureConfig = namedtuple(
#     'FeatureConfig',
#     [
#         'target',
#         'index_cols'
#         'time_varying_cols',

#     ],
# )


def format_plot(fig, legends = None, xlabel="Time", ylabel="Value", title="", font_size=15):
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_layout(
            autosize=False,
            width=900,
            height=500,
            title_text=title,
            title={
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            titlefont={
                "size": 20
            },
            legend_title = None,
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
            )
        )
    return fig

def eval_model(model, ts_train, ts_test, name=None):
    if name is None:
        name = type(model).__name__
    model.fit(ts_train)
    y_pred = model.predict(len(ts_test))
    return y_pred, {
        "Algorithm": name,
        "MAE": mae(actual_series = ts_test, pred_series = y_pred),
        "MSE": mse(actual_series = ts_test, pred_series = y_pred),
        "MASE": mase(actual_series = ts_test, pred_series = y_pred, insample=ts_train),
        "Forecast Bias": forecast_bias(actual_series = ts_test, pred_series = y_pred)
    }

def format_y_pred(y_pred, name):
    y_pred = y_pred.data_array().to_series()
    y_pred.index = y_pred.index.get_level_values(0)
    y_pred.name = name
    return y_pred

def plot_forecast(pred_df, forecast_columns, forecast_display_names=None):
    if forecast_display_names is None:
        forecast_display_names = forecast_columns
    else:
        assert len(forecast_columns)==len(forecast_display_names)
    mask = ~pred_df[forecast_columns[0]].isnull()
    # colors = ["rgba("+",".join([str(c) for c in plotting_utils.hex_to_rgb(c)])+",<alpha>)" for c in px.colors.qualitative.Plotly]
    colors = [c.replace("rgb", "rgba").replace(")", ", <alpha>)") for c in px.colors.qualitative.Dark2]
    # colors = [c.replace("rgb", "rgba").replace(")", ", <alpha>)") for c in px.colors.qualitative.Safe]
    act_color = colors[0]
    colors = cycle(colors[1:])
    dash_types = cycle(["dash","dot","dashdot"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df[mask].index, y=pred_df[mask].covidOccupiedMVBeds,
                        mode='lines', line = dict(color=act_color.replace("<alpha>", "0.3")),
                        name='Actual Data'))
    for col, display_col in zip(forecast_columns,forecast_display_names):
        fig.add_trace(go.Scatter(x=pred_df[mask].index, y=pred_df.loc[mask, col],
                            mode='lines', line = dict(dash=next(dash_types), color=next(colors).replace("<alpha>", "1")),
                            name=display_col))
    return fig

train_df = pd.read_pickle(preprocessed / 'East of England_train.pkl')
val_df = pd.read_pickle(preprocessed / 'East of England_val.pkl')
test_df = pd.read_pickle(preprocessed / 'East of England_test.pkl')

# # 7 days moving average for the train, val and test data
# train_df['covidOccupiedMVBeds'] = train_df['covidOccupiedMVBeds'].rolling(7).mean()
# val_df['covidOccupiedMVBeds'] = val_df['covidOccupiedMVBeds'].rolling(7).mean()
# test_df['covidOccupiedMVBeds'] = test_df['covidOccupiedMVBeds'].rolling(7).mean()

# # clean the train, val and test data
# train_df = train_df.dropna()
# val_df = val_df.dropna()
# test_df = test_df.dropna()

# select date and covidOccupiedMVBeds
freq = train_df.iloc[0]['covidOccupiedMVBeds']
train_df = train_df[['date', 'covidOccupiedMVBeds']].set_index('date')
val_df = val_df[['date', 'covidOccupiedMVBeds']].set_index('date')
test_df = test_df[['date', 'covidOccupiedMVBeds']].set_index('date')

pred_df = pd.concat([train_df, val_df, test_df], axis=0)
metric_record = []

from src.transforms.target_transformations import AutoStationaryTransformer
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from plotly.subplots import make_subplots

transformer_pipeline = {}

transformer_pipeline['covidOccupiedMVBeds'] = AutoStationaryTransformer()
# train_df = transformer_pipeline['covidOccupiedMVBeds'].fit_transform(train_df, freq='D')


# Decompose the time series to analyze the trend, seasonality and residuals on the pred_df
decompose = seasonal_decompose(pred_df, model='additive', period=7)

# plot the decomposed time series
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1)

fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df.covidOccupiedMVBeds, mode='lines', name='Actual Data'), row=1, col=1)
fig.add_trace(go.Scatter(x=pred_df.index, y=decompose.trend, mode='lines', name='Trend'), row=1, col=1)
fig.add_trace(go.Scatter(x=pred_df.index, y=decompose.seasonal, mode='lines', name='Seasonal'), row=2, col=1)
fig.add_trace(go.Scatter(x=pred_df.index, y=decompose.resid, mode='lines', name='Residual'), row=3, col=1)

fig.update_layout(height=900, width=900, title_text="Decomposition of the Time Series")

fig.update_xaxes(title_text="Date", row=3, col=1)
fig.update_yaxes(title_text="Trend", row=1, col=1)
fig.update_yaxes(title_text="Seasonal", row=2, col=1)
fig.update_yaxes(title_text="Residual", row=3, col=1)

fig.show()

# Acf and Pacf plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, ax = plt.subplots(2, 1, figsize=(15, 10))
plot_acf(pred_df.covidOccupiedMVBeds, lags=30, ax=ax[0])
plot_pacf(pred_df.covidOccupiedMVBeds, lags=30, ax=ax[1])
plt.show()



ts_train = TimeSeries.from_series(train_df)
ts_val = TimeSeries.from_series(val_df)
ts_test = TimeSeries.from_series(test_df)

# base line
name = "ARIMA"

model = ARIMA(p=1, d=1, q=1, seasonal_order=(1, 1, 1, 7 ))

with LogTime() as It:
    y_pred, metrics = eval_model(model, ts_train, ts_val, name)
metrics['Time elapsed'] = It.elapsed
metric_record.append(metrics)
y_pred = format_y_pred(y_pred, 'ARIMA prediction')
pred_df = pred_df.join(y_pred)

fig = plot_forecast(pred_df, ['ARIMA prediction'], ['ARIMA prediction'])
fig = format_plot(fig, title=f"ARIMA: MAE={metrics['MAE']:.2f} | MSE={metrics['MSE']:.2f} | MASE={metrics['MASE']:.2f} | Forecast Bias={metrics['Forecast Bias']:.2f}")
fig.update_xaxes(type='date', rangeslider_visible=True)
fig.show()