import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Set matplotlib style and parameters
plt.style.use("seaborn-v0_8-poster")
plt.rcParams.update(
    {
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
    }
)

# Device setup for CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def load_and_preprocess_data(
    filepath,
    areaname,
    recovery_period=16,
    rolling_window=7,
    start_date="2020-04-01",
    end_date="2020-12-31",
):
    """Load and preprocess the data from a CSV file."""
    df = pd.read_csv(filepath)
    df = df[df["areaName"] == areaname].reset_index(drop=True)
    df = df[::-1].reset_index(drop=True)  # Reverse dataset if needed

    df["date"] = pd.to_datetime(df["date"])
    df = df[
        (df["date"] >= pd.to_datetime(start_date))
        & (df["date"] <= pd.to_datetime(end_date))
    ]

    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df[
        "cumulative_deceased"
    ].shift(recovery_period)
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)
    df["active_cases"] = (
        df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]
    )
    df["susceptible"] = df["population"] - (
        df["recovered"] + df["cumulative_deceased"] + df["active_cases"]
    )
    df["exposed"] = 1.1 * df["active_cases"].shift(1).fillna(0)
    df["exposed"] = df["exposed"].clip(lower=0)

    cols_to_smooth = [
        "susceptible",
        "exposed",
        "cumulative_confirmed",
        "cumulative_deceased",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "recovered",
        "active_cases",
    ]
    for col in cols_to_smooth:
        if col in df.columns:
            df[col] = (
                df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)
            )

    return df


# Load and preprocess the data
data = load_and_preprocess_data(
    "../../data/processed/merged_nhs_covid_data.csv",
    areaname="London",
    recovery_period=21,
    start_date="2020-05-01",
    end_date="2021-05-31",
).drop(columns=["Unnamed: 0"], axis=1)

areaname="London"

# plot suscepitble data over time to check for any trends
plt.figure()
plt.plot(data["date"], data["recovered"], label="recovered")
plt.xlabel("Date")
plt.ylabel("recovered")
plt.title(f"recovered Over Time in {areaname}")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig(f"../../reports/figures/{areaname}_susceptible.pdf")
plt.show()

class SEIRDNet(nn.Module):
    """Epidemiological network for predicting SEIRD model outputs."""

    def __init__(
        self,
        inverse=False,
        init_beta=None,
        init_gamma=None,
        init_delta=None,
        retain_seed=42,
        num_layers=4,
        hidden_neurons=20,
    ):
        super(SEIRDNet, self).__init__()
        self.retain_seed = retain_seed
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(
            nn.Linear(hidden_neurons, 5)
        )  # Adjust the output size to 5 (S, E, I, R, D)
        self.net = nn.Sequential(*layers)

        if inverse:
            self._beta = nn.Parameter(
                torch.tensor(
                    [init_beta if init_beta is not None else torch.rand(1)],
                    device=device,
                ),
                requires_grad=True,
            )
            self._gamma = nn.Parameter(
                torch.tensor(
                    [init_gamma if init_gamma is not None else torch.rand(1)],
                    device=device,
                ),
                requires_grad=True,
            )
            self._delta = nn.Parameter(
                torch.tensor(
                    [init_delta if init_delta is not None else torch.rand(1)],
                    device=device,
                ),
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
                    m.bias.data.fill_(0.001)

        self.apply(init_weights)


def network_prediction(t, model, device, scaler, N):
    """
    Generate predictions from the SEIRDNet model.

    Parameters:
    t (numpy array): Time input.
    model (SEIRDNet): Trained SEIRDNet model.
    device (torch.device): Device to run the model on.
    scaler (MinMaxScaler): Scaler used to normalize the data.
    N (float): Population size.

    Returns:
    np.array: Scaled predictions from the model.
    """
    # Convert time input to tensor and move to the appropriate device
    t_tensor = torch.from_numpy(t).float().view(-1, 1).to(device).requires_grad_(True)

    # Disable gradient computation for prediction
    with torch.no_grad():
        # Generate predictions from the model
        predictions = model(t_tensor)

        # Convert predictions to numpy array
        predictions = predictions.cpu().numpy()

        # Inverse transform the predictions to original scale
        predictions = scaler.inverse_transform(predictions)

    return predictions


def SEIRD_model(u, t, beta, sigma, gamma, delta, N):
    S, E, I, R, D = u
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (gamma + delta) * I
    dRdt = gamma * I
    dDdt = delta * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]


# Prepare PyTorch tensors from the data
def prepare_tensors(data, device):
    """Prepare tensors for training."""
    t = (
        tensor(np.arange(1, len(data) + 1), dtype=torch.float32)
        .view(-1, 1)
        .to(device)
        .requires_grad_(True)
    )
    S = tensor(data["susceptible"].values, dtype=torch.float32).view(-1, 1).to(device)
    E = tensor(data["exposed"].values, dtype=torch.float32).view(-1, 1).to(device)
    I = tensor(data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
    R = tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = (
        tensor(data["cumulative_deceased"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    return t, S, E, I, R, D


# Split and scale the data into training and validation sets
def split_and_scale_data(data, train_size, features, device):
    """Split and scale data into training and validation sets."""
    scaler = MinMaxScaler()
    scaler.fit(data[features])

    # Select the first train_size days for training
    train_data = data.iloc[:train_size]

    # Select the remaining days for validation
    val_data = data.iloc[train_size:]

    # Scale the data
    scaled_train_data = pd.DataFrame(
        scaler.transform(train_data[features]), columns=features
    )
    scaled_val_data = pd.DataFrame(
        scaler.transform(val_data[features]), columns=features
    )

    # Prepare tensors for each segment
    t_train, S_train, E_train, I_train, R_train, D_train = prepare_tensors(
        scaled_train_data, device
    )
    t_val, S_val, E_val, I_val, R_val, D_val = prepare_tensors(scaled_val_data, device)

    tensor_data = {
        "train": (t_train, S_train, E_train, I_train, R_train, D_train),
        "val": (t_val, S_val, E_val, I_val, R_val, D_val),
    }

    return tensor_data, scaler


# Example features and data split
features = [
    "susceptible",
    "exposed",
    "active_cases",
    "recovered",
    "cumulative_deceased",
]

# Set the training size to 90% of the data
train_size = int(0.9 * len(data))

tensor_data, scaler = split_and_scale_data(data, train_size, features, device)


# PINN loss function
def pinn_loss(
    tensor_data,
    parameters,
    model_output,
    t,
    N,
    sigma=1 / 5,
    beta=None,
    gamma=None,
    delta=None,
    train_size=None,
):
    """Physics-Informed Neural Network loss function."""
    S_pred, E_pred, I_pred, R_pred, D_pred = torch.split(model_output, 1, dim=1)

    S_train, E_train, I_train, R_train, D_train = tensor_data["train"][1:]
    S_val, E_val, I_val, R_val, D_val = tensor_data["val"][1:]

    S_total = torch.cat([S_train, S_val])
    E_total = torch.cat([E_train, E_val])
    I_total = torch.cat([I_train, I_val])
    R_total = torch.cat([R_train, R_val])
    D_total = torch.cat([D_train, D_val])

    s_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    e_t = grad(E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True)[0]
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
    dEdt = beta * S_pred * I_pred / N - sigma * E_pred
    dIdt = sigma * E_pred - (gamma + delta) * I_pred
    dRdt = gamma * I_pred
    dDdt = delta * I_pred

    # Generate a random subset of indices
    if train_size is not None:
        index = torch.randperm(train_size)
    else:
        index = torch.arange(len(t))

    # Data fitting loss for the random subset of indices
    data_fitting_loss = (
        torch.mean((S_pred[index] - S_total[index]) ** 2)
        + torch.mean((E_pred[index] - E_total[index]) ** 2)
        + torch.mean((I_pred[index] - I_total[index]) ** 2)
        + torch.mean((R_pred[index] - R_total[index]) ** 2)
        + torch.mean((D_pred[index] - D_total[index]) ** 2)
    )

    # Differential equation residuals
    residual_loss = (
        torch.mean((s_t - dSdt) ** 2)
        + torch.mean((e_t - dEdt) ** 2)
        + torch.mean((i_t - dIdt) ** 2)
        + torch.mean((r_t - dRdt) ** 2)
        + torch.mean((d_t - dDdt) ** 2)
    )

    # Initial condition loss
    S0, E0, I0, R0, D0 = S_train[0], E_train[0], I_train[0], R_train[0], D_train[0]
    initial_condition_loss = (
        (S_pred[0] - S0) ** 2
        + (E_pred[0] - E0) ** 2
        + (I_pred[0] - I0) ** 2
        + (R_pred[0] - R0) ** 2
        + (D_pred[0] - D0) ** 2
    )

    # Weights for the loss terms
    lambda_data = 1.0
    lambda_residual = 1.0
    lambda_initial = 1.0

    # Total loss
    total_loss = (
        lambda_data * data_fitting_loss
        + lambda_residual * residual_loss
        + lambda_initial * initial_condition_loss
    )

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


# Initialize parameters and model
N = data["population"].values[0]
model = SEIRDNet(
    inverse=True,
    init_beta=0.3,
    init_gamma=0.1,
    init_delta=0.1,
    num_layers=10,
    hidden_neurons=32,
    retain_seed=100,
).to(device)

# Initialize optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=5000, gamma=0.9)

# Initialize early stopping
earlystopping = EarlyStopping(patience=100, verbose=False)

# Set the number of epochs for training
epochs = 100000

# Full time input for the entire dataset
t = (
    torch.tensor(np.arange(len(data)), dtype=torch.float32)
    .view(-1, 1)
    .to(device)
    .requires_grad_(True)
)

# Shuffle the data index
index = torch.randperm(len(tensor_data["train"][0]))

# List to store loss history
loss_history = []


# Training loop function
def train_loop(
    model,
    optimizer,
    scheduler,
    earlystopping,
    epochs,
    tensor_data,
    N,
    loss_history,
    index,
):
    """Training loop for the model."""
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()

        # Shuffle the training index for each epoch
        index = torch.randperm(len(tensor_data["train"][0]))

        # Get the model output
        model_output = model(t)

        # Compute the loss
        loss = pinn_loss(
            tensor_data,
            model,
            model_output,
            t,
            N,
            train_size=len(tensor_data["train"][0]),
        )

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Store the loss
        loss_history.append(loss.item())

        # Early stopping
        earlystopping(loss.item())
        if earlystopping.early_stop:
            print("Early stopping")
            break

        # Print progress every 100 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

    return model, loss_history


# Train the model
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

# Plot the loss history
plt.figure()
plt.plot(np.log10(loss_history), label="Training Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Log10(Loss)")
plt.title("Training Loss History")
plt.legend()
plt.show()

# Generate predictions for the entire dataset
t_values = np.arange(len(data))
predictions = network_prediction(t_values, model, device, scaler, N)

dates = data["date"]

# Extract predictions for each compartment
S_pred = predictions[:, 0]
E_pred = predictions[:, 1]
I_pred = predictions[:, 2]
R_pred = predictions[:, 3]
D_pred = predictions[:, 4]

# Actual data
S_actual = data["susceptible"].values
E_actual = data["exposed"].values
I_actual = data["active_cases"].values
R_actual = data["recovered"].values
D_actual = data["cumulative_deceased"].values

# Define training index size
train_index_size = len(tensor_data["train"][0])

# Plot the results
# fig, ax = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

# # Plot the susceptible compartment
# ax[0].plot(dates, S_actual, label="True Susceptible", color="blue")
# ax[0].plot(dates, S_pred, label="Predicted Susceptible", color="red", linestyle="--")
# # ax[0].scatter(dates[:train_index_size], S_actual[:train_index_size], color="green", label="Train-Val Split")
# # dotted line for the training-validation split
# ax[0].axvline(
#     x=dates[train_index_size],
#     color="black",
#     linestyle="--",
#     linewidth=1,
#     label="Train-Val Split",
# )
# ax[0].set_ylabel("Susceptible")
# ax[0].legend()

# # Plot the exposed compartment
# ax[1].plot(dates, E_actual, label="True Exposed", color="blue")
# ax[1].plot(dates, E_pred, label="Predicted Exposed", color="red", linestyle="--")
# # ax[1].scatter(dates[:train_index_size], E_actual[:train_index_size], color="green", label="Train-Val Split")
# ax[1].axvline(
#     x=dates[train_index_size],
#     color="black",
#     linestyle="--",
#     linewidth=1,
#     label="Train-Val Split",
# )
# ax[1].set_ylabel("Exposed")
# ax[1].legend()

# # Plot the active cases compartment
# ax[2].plot(dates, I_actual, label="True Active Cases", color="blue")
# ax[2].plot(dates, I_pred, label="Predicted Active Cases", color="red", linestyle="--")
# # ax[2].scatter(dates[:train_index_size], I_actual[:train_index_size], color="green", label="Train-Val Split")
# ax[2].axvline(
#     x=dates[train_index_size],
#     color="black",
#     linestyle="--",
#     linewidth=1,
#     label="Train-Val Split",
# )
# ax[2].set_ylabel("Active Cases")
# ax[2].legend()

# # Plot the recovered compartment
# ax[3].plot(dates, R_actual, label="True Recovered", color="blue")
# ax[3].plot(dates, R_pred, label="Predicted Recovered", color="red", linestyle="--")
# # ax[3].scatter(dates[:train_index_size], R_actual[:train_index_size], color="green", label="Train-Val Split")
# ax[3].axvline(
#     x=dates[train_index_size],
#     color="black",
#     linestyle="--",
#     linewidth=1,
#     label="Train-Val Split",
# )
# ax[3].set_ylabel("Recovered")
# ax[3].legend()

# # Plot the deceased compartment
# ax[4].plot(dates, D_actual, label="True Deceased", color="blue")
# ax[4].plot(dates, D_pred, label="Predicted Deceased", color="red", linestyle="--")
# # ax[4].scatter(dates[:train_index_size], D_actual[:train_index_size], color="green", label="Train-Val Split")
# ax[4].axvline(
#     x=dates[train_index_size],
#     color="black",
#     linestyle="--",
#     linewidth=1,
#     label="Train-Val Split",
# )
# ax[4].set_ylabel("Deceased")
# ax[4].legend()

# plt.xlabel("Date")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig(f"../../reports/figures/pinn_{areaname}_results.pdf")
# plt.show()

import matplotlib.dates as mdates

fig, ax = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
# Define plot details
plot_details = [
    ("Susceptible", S_actual, S_pred),
    ("Exposed", E_actual, E_pred),
    ("Active Cases", I_actual, I_pred),
    ("Recovered", R_actual, R_pred),
    ("Deceased", D_actual, D_pred),
]

for i, (title, actual, pred) in enumerate(plot_details):
    ax[i].plot(dates, actual, label="True", color="blue", linewidth=2)
    ax[i].plot(dates, pred, label="Predicted", color="red", linestyle="--", linewidth=2)
    ax[i].axvline(
        x=dates[train_index_size],
        color="black",
        linestyle="--",
        linewidth=1,
        label="Train-test Split",
    )
    ax[i].set_ylabel(title, fontsize=14)
    ax[i].grid(True)
    ax[i].set_title(title + " Over Time", fontsize=16)
    ax[i].tick_params(axis="both", which="major", labelsize=12)

# Only add the legend to the first subplot
handles, labels = ax[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="upper center", fontsize=12, ncol=3, bbox_to_anchor=(0.5, 1.05)
)

ax[-1].set_xlabel("Date", fontsize=14)
ax[-1].xaxis.set_major_locator(mdates.MonthLocator())
ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45, ha="right")
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the legend
plt.subplots_adjust(top=0.9)
plt.savefig(f"../../reports/figures/pinn_{areaname}_results.pdf")
plt.show()

# Extract the parameter values
beta = model.beta.item()
gamma = model.gamma.item()
delta = model.delta.item()

# Print the parameter values
print(f"Estimated beta: {beta:.4f}")
print(f"Estimated gamma: {gamma:.4f}")
print(f"Estimated delta: {delta:.4f}")

# save the output
output = pd.DataFrame(
    {
        "date": dates,
        "susceptible": S_pred,
        "exposed": E_pred,
        "active_cases": I_pred,
        "recovered": R_pred,
        "cumulative_deceased": D_pred,
    }
)

output.to_csv(f"../../reports/output/pinn_{areaname}_output.csv", index=False)
