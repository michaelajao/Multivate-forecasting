import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm.notebook import tqdm
from scipy.integrate import odeint
from collections import deque
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from torch import tensor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set matplotlib style and parameters
plt.style.use("seaborn-v0_8-poster")
plt.rcParams.update({
    "font.size": 20,
    "figure.figsize": [10, 5],
    "figure.facecolor": "white",
    "figure.autolayout": True,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.formatter.limits": (0, 5),
    "axes.formatter.use_mathtext": True,
    "axes.formatter.useoffset": False,
    "axes.xmargin": 0,
    "axes.ymargin": 0,
    "legend.fontsize": 14,
    "legend.frameon": False,
    "legend.loc": "best",
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "xtick.labelsize": 14,
    "xtick.direction": "in",
    "xtick.top": False,
    "ytick.labelsize": 14,
    "ytick.direction": "in",
    "ytick.right": False,
    "grid.color": "grey",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "errorbar.capsize": 4,
    "figure.subplot.wspace": 0.4,
    "figure.subplot.hspace": 0.4,
    "image.cmap": "viridis",
})

# Device setup for CUDA or CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(seed)


def check_pytorch():
    """Check PyTorch and CUDA setup."""
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available. PyTorch will run on CPU.")


check_pytorch()


def load_and_preprocess_data(filepath, recovery_period=16, rolling_window=7, start_date="2020-04-01", end_date="2020-05-31"):
    """Load and preprocess the data from a CSV file."""
    df = pd.read_csv(filepath)
    
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    
    # Calculate recovered cases based on shifted cumulative confirmed and deceased cases
    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df["cumulative_deceased"].shift(recovery_period)
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)
    df["active_cases"] = df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]
    df["active_cases"] = df["active_cases"].clip(lower=0)
    df["susceptible"] = df["population"] - (df["recovered"] + df["cumulative_deceased"] + df["active_cases"])

    # Columns to apply rolling mean
    cols_to_smooth = [
        "susceptible",
        "new_confirmed",
        "cumulative_confirmed",
        "cumulative_deceased",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "recovered",
        "new_deceased",
        "active_cases",
    ]
    
    # Apply rolling mean to smooth the data
    for col in cols_to_smooth:
        if col in df.columns:
            df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    return df


# Load and preprocess the data
data = load_and_preprocess_data(
    "../../data/hos_data/england_data.csv",
    recovery_period=21,
    rolling_window=7,
    start_date="2020-05-01",
    end_date="2020-12-31",
)

areaname = "England"

# Plot recovered data over time to check for any trends
plt.figure()
plt.plot(data["date"], data["recovered"], label="recovered")
plt.xlabel("Date")
plt.ylabel("Recovered")
plt.title(f"Recovered Over Time in {areaname}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


class SEIRDNet(nn.Module):
    """Epidemiological network for predicting SEIRD model outputs."""

    def __init__(self, inverse=False, init_beta=None, init_gamma=None, init_delta=None, retain_seed=42, num_layers=4, hidden_neurons=20):
        super(SEIRDNet, self).__init__()
        self.retain_seed = retain_seed
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 4))  # Adjust the output size to 4 (S, I, R, D)
        self.net = nn.Sequential(*layers)

        if inverse:
            self._beta = nn.Parameter(
                torch.tensor([init_beta if init_beta is not None else torch.rand(1)], device=device),
                requires_grad=True,
            )
            self._gamma = nn.Parameter(
                torch.tensor([init_gamma if init_gamma is not None else torch.rand(1)], device=device),
                requires_grad=True,
            )
            self._delta = nn.Parameter(
                torch.tensor([init_delta if init_delta is not None else torch.rand(1)], device=device),
                requires_grad=True,
            )
        else:
            self._beta = None
            self._gamma = None
            self._delta = None

        self.init_xavier()

    def forward(self, t):
        return self.net(t)

    @property
    def beta(self):
        return torch.sigmoid(self._beta) if self._beta is not None else None

    @property
    def gamma(self):
        return torch.sigmoid(self._gamma) if self._gamma is not None else None

    @property
    def delta(self):
        return torch.sigmoid(self._delta) if self._delta is not None else None

    def init_xavier(self):
        torch.manual_seed(self.retain_seed)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(init_weights)


def network_prediction(t, model, device, scaler, N):
    """Generate predictions from the SEIRDNet model."""
    t_tensor = torch.from_numpy(t).float().view(-1, 1).to(device).requires_grad_(True)

    with torch.no_grad():
        predictions = model(t_tensor).cpu().numpy()
        predictions = scaler.inverse_transform(predictions)

    return predictions


def SIRD_model(y, t, beta, gamma, delta, N):
    S, I, R, D = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - (gamma + delta) * I
    dRdt = gamma * I
    dDdt = delta * I
    return [dSdt, dIdt, dRdt, dDdt]


def prepare_tensors(data, device):
    """Prepare tensors for training."""
    t = tensor(np.arange(1, len(data) + 1), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
    S = tensor(data["susceptible"].values, dtype=torch.float32).view(-1, 1).to(device)
    I = tensor(data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
    R = tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = tensor(data["cumulative_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
    return t, S, I, R, D


def split_and_scale_data(data, train_size, features, device):
    """Split and scale data into training and validation sets."""
    scaler = MinMaxScaler()
    scaler.fit(data[features])

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:]

    scaled_train_data = pd.DataFrame(scaler.transform(train_data[features]), columns=features)
    scaled_val_data = pd.DataFrame(scaler.transform(val_data[features]), columns=features)

    t_train, S_train, I_train, R_train, D_train = prepare_tensors(scaled_train_data, device)
    t_val, S_val, I_val, R_val, D_val = prepare_tensors(scaled_val_data, device)

    tensor_data = {
        "train": (t_train, S_train, I_train, R_train, D_train),
        "val": (t_val, S_val, I_val, R_val, D_val),
    }

    return tensor_data, scaler


# Example features and data split
features = ["susceptible", "active_cases", "recovered", "cumulative_deceased"]

# Set the training size to 90% of the data
train_size = 200

tensor_data, scaler = split_and_scale_data(data, train_size, features, device)


def pinn_loss(tensor_data, parameters, model_output, t, N, sigma=1 / 5, beta=None, gamma=None, delta=None, train_size=None):
    """Physics-Informed Neural Network loss function."""
    S_pred, I_pred, R_pred, D_pred = torch.split(model_output, 1, dim=1)

    S_train, I_train, R_train, D_train = tensor_data["train"][1:]
    S_val, I_val, R_val, D_val = tensor_data["val"][1:]

    S_total = torch.cat([S_train, S_val])
    I_total = torch.cat([I_train, I_val])
    R_total = torch.cat([R_train, R_val])
    D_total = torch.cat([D_train, D_val])

    s_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    i_t = grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    r_t = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    d_t = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]

    if beta is None:
        beta = parameters.beta
    if gamma is None:
        gamma = parameters.gamma
    if delta is None:
        delta = parameters.delta

    dSdt = -beta * S_pred * I_pred / N
    dIdt = beta * S_pred * I_pred / N - (gamma + delta) * I_pred
    dRdt = gamma * I_pred
    dDdt = delta * I_pred

    if train_size is not None:
        index = torch.randperm(train_size)
    else:
        index = torch.arange(len(t))

    data_fitting_loss = (
        torch.mean((S_pred[index] - S_total[index]) ** 2)
        + torch.mean((I_pred[index] - I_total[index]) ** 2)
        + torch.mean((R_pred[index] - R_total[index]) ** 2)
        + torch.mean((D_pred[index] - D_total[index]) ** 2)
    )

    residual_loss = (
        torch.mean((s_t - dSdt) ** 2)
        + torch.mean((i_t - dIdt) ** 2)
        + torch.mean((r_t - dRdt) ** 2)
        + torch.mean((d_t - dDdt) ** 2)
    )

    S0, I0, R0, D0 = S_train[0], I_train[0], R_train[0], D_train[0]
    initial_condition_loss = (
        torch.mean((S_pred[0] - S0) ** 2)
        + torch.mean((I_pred[0] - I0) ** 2)
        + torch.mean((R_pred[0] - R0) ** 2)
        + torch.mean((D_pred[0] - D0) ** 2)
    )

    total_loss = data_fitting_loss + residual_loss + initial_condition_loss

    return total_loss


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0
        self.loss_history = deque(maxlen=patience + 1)

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


N = data["population"].values[0]
model = SEIRDNet(
    inverse=True,
    init_beta=0.1,
    init_gamma=0.1,
    init_delta=0.1,
    num_layers=6,
    hidden_neurons=32,
    retain_seed=100,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=10000, gamma=0.1)
earlystopping = EarlyStopping(patience=100, verbose=False)
epochs = 100000

t = torch.tensor(np.arange(len(data)), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
index = torch.randperm(len(tensor_data["train"][0]))
loss_history = []


def train_loop(model, optimizer, scheduler, earlystopping, epochs, tensor_data, N, loss_history, index):
    """Training loop for the model."""
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()

        index = torch.randperm(len(tensor_data["train"][0]))

        model_output = model(t)

        loss = pinn_loss(
            tensor_data,
            model,
            model_output,
            t,
            N,
            train_size=len(index),
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        earlystopping(loss.item())
        if earlystopping.early_stop:
            print("Early stopping")
            break

        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    return model, loss_history


model, loss_history = train_loop(
    model,
    optimizer,
    scheduler,
    earlystopping,
    epochs,
    tensor_data,
    N,
    loss_history,
    index,
)

plt.figure()
plt.plot(np.log10(loss_history), label="Training Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Log10(Loss)")
plt.title("Training Loss History")
plt.legend()
plt.show()

t_values = np.arange(len(data))
predictions = network_prediction(t_values, model, device, scaler, N)

dates = data["date"]

S_pred = predictions[:, 0]
I_pred = predictions[:, 1]
R_pred = predictions[:, 2]
D_pred = predictions[:, 3]

S_actual = data["susceptible"].values
I_actual = data["active_cases"].values
R_actual = data["recovered"].values
D_actual = data["cumulative_deceased"].values

train_index_size = len(tensor_data["train"][0])

fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
plot_details = [
    ("Susceptible", S_actual, S_pred),
    ("Active Cases", I_actual, I_pred),
    ("Recovered", R_actual, R_pred),
    ("Deceased", D_actual, D_pred),
]

for i, (title, actual, pred) in enumerate(plot_details):
    ax[i].scatter(dates, actual, label="True data", color="blue", s=10, alpha=0.6)
    ax[i].plot(dates, pred, label="Predicted data", color="red")
    ax[i].axvline(
        x=dates[train_index_size],
        color="black",
        linestyle="--",
        linewidth=2,
        label="Train_size",
    )
    ax[i].set_ylabel(title, fontsize=14)
    ax[i].grid(True)
    # ax[i].set_title(title + " Over Time", fontsize=16)
    ax[i].tick_params(axis="both", which="major", labelsize=12)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="upper center", fontsize=12, ncol=3, bbox_to_anchor=(0.5, 1.05)
)

ax[-1].set_xlabel("Date", fontsize=14)
ax[-1].xaxis.set_major_locator(mdates.MonthLocator())
ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45, ha="right")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(top=0.7)
plt.savefig(f"../../reports/figures/{train_size}_pinn_{areaname}_results.pdf")
plt.show()

beta = model.beta.item()
gamma = model.gamma.item()
delta = model.delta.item()

print(f"Estimated beta: {beta:.4f}")
print(f"Estimated gamma: {gamma:.4f}")
print(f"Estimated delta: {delta:.4f}")

output = pd.DataFrame(
    {
        "date": dates,
        "susceptible": S_pred,
        "active_cases": I_pred,
        "recovered": R_pred,
        "cumulative_deceased": D_pred,
    }
)

output.to_csv(f"../../reports/output/{train_size}_pinn_{areaname}_output.csv", index=False)
torch.save(model.state_dict(), f"../../models/{train_size}_pinn_{areaname}_model.pth")


def mean_absolute_scaled_error(y_true, y_pred, benchmark=None):
    """Calculate the Mean Absolute Scaled Error (MASE)."""
    if benchmark is None:
        benchmark = np.roll(y_true, 1)
        benchmark[0] = y_true[0]

    mae_benchmark = mean_absolute_error(y_true, benchmark)
    mae_model = mean_absolute_error(y_true, y_pred)

    return mae_model / mae_benchmark


def forecast_bias(y_true, y_pred):
    """Calculate the forecast bias."""
    return np.mean(y_pred - y_true)


def evaluate_model(model, data, scaler, device, N):
    """Evaluate the trained model on the dataset and calculate evaluation metrics."""
    model.eval()
    with torch.no_grad():
        t_values = np.arange(len(data))
        predictions = network_prediction(t_values, model, device, scaler, N)

        S_pred = predictions[:, 0]
        I_pred = predictions[:, 1]
        R_pred = predictions[:, 2]
        D_pred = predictions[:, 3]

        S_actual = data["susceptible"].values
        I_actual = data["active_cases"].values
        R_actual = data["recovered"].values
        D_actual = data["cumulative_deceased"].values

        mae_s = mean_absolute_error(S_actual, S_pred)
        mae_i = mean_absolute_error(I_actual, I_pred)
        mae_r = mean_absolute_error(R_actual, R_pred)
        mae_d = mean_absolute_error(D_actual, D_pred)

        mse_s = mean_squared_error(S_actual, S_pred)
        mse_i = mean_squared_error(I_actual, I_pred)
        mse_r = mean_squared_error(R_actual, R_pred)
        mse_d = mean_squared_error(D_actual, D_pred)

        mase_s = mean_absolute_scaled_error(S_actual, S_pred)
        mase_i = mean_absolute_scaled_error(I_actual, I_pred)
        mase_r = mean_absolute_scaled_error(R_actual, R_pred)
        mase_d = mean_absolute_scaled_error(D_actual, D_pred)

        bias_s = forecast_bias(S_actual, S_pred)
        bias_i = forecast_bias(I_actual, I_pred)
        bias_r = forecast_bias(R_actual, R_pred)
        bias_d = forecast_bias(D_actual, D_pred)

        results = {
            "MAE_Susceptible": mae_s,
            "MAE_Infected": mae_i,
            "MAE_Recovered": mae_r,
            "MAE_Deceased": mae_d,
            "MSE_Susceptible": mse_s,
            "MSE_Infected": mse_i,
            "MSE_Recovered": mse_r,
            "MSE_Deceased": mse_d,
            "MASE_Susceptible": mase_s,
            "MASE_Infected": mase_i,
            "MASE_Recovered": mase_r,
            "MASE_Deceased": mase_d,
            "Forecast_Bias_Susceptible": bias_s,
            "Forecast_Bias_Infected": bias_i,
            "Forecast_Bias_Recovered": bias_r,
            "Forecast_Bias_Deceased": bias_d,
        }

        return results


results = evaluate_model(model, data, scaler, device, N)
print(results)

results_df = pd.DataFrame(results, index=[0])
results_df.to_csv(f"../../reports/results/{train_size}_pinn_{areaname}_results.csv", index=False)
