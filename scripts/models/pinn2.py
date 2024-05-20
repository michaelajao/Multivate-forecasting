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

def load_and_preprocess_data(filepath, areaname, recovery_period=16, rolling_window=7, start_date="2020-04-01", end_date="2020-12-31"):
    """Load and preprocess the data from a CSV file."""
    df = pd.read_csv(filepath)
    df = df[df["areaName"] == areaname].reset_index(drop=True)
    df = df[::-1].reset_index(drop=True)  # Reverse dataset if needed

    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df["cumulative_deceased"].shift(recovery_period)
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)
    df["active_cases"] = df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]

    cols_to_smooth = ["cumulative_confirmed", "cumulative_deceased", "hospitalCases", "covidOccupiedMVBeds", "recovered", "active_cases"]
    for col in cols_to_smooth:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    return df

# Load and preprocess the data
data = load_and_preprocess_data("../../data/hos_data/merged_data.csv", areaname="South West", recovery_period=21, start_date="2020-04-01", end_date="2021-12-31").drop(columns=["Unnamed: 0"], axis=1)

class SEIRDNet(nn.Module):
    """Epidemiological network for predicting SEIRD model outputs."""
    def __init__(self, inverse=False, init_beta=None, init_gamma=None, init_delta=None, retain_seed=42, num_layers=4, hidden_neurons=20):
        super(SEIRDNet, self).__init__()
        self.retain_seed = retain_seed
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 5))  # Adjust the output size to 5 (S, E, I, R, D)
        self.net = nn.Sequential(*layers)
        
        if inverse:
            self._beta = nn.Parameter(torch.tensor([init_beta if init_beta is not None else torch.rand(1)], device=device), requires_grad=True)
            self._gamma = nn.Parameter(torch.tensor([init_gamma if init_gamma is not None else torch.rand(1)], device=device), requires_grad=True)
            self._delta = nn.Parameter(torch.tensor([init_delta if init_delta is not None else torch.rand(1)], device=device), requires_grad=True)
        else:
            self._beta = None
            self._gamma = None
            self._delta = None
        
        self.init_xavier()

    def forward(self, t):
        return self.net(t)

    @property
    def beta(self):
        return torch.sigmoid(self._beta) * 0.9 + 0.1 if self._beta is not None else None

    @property
    def gamma(self):
        return torch.sigmoid(self._gamma) * 0.09 + 0.01 if self._gamma is not None else None
    
    @property
    def delta(self):
        return torch.sigmoid(self._delta) * 0.09 + 0.01 if self._delta is not None else None
    
    def init_xavier(self):
        torch.manual_seed(self.retain_seed)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(init_weights)

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
    t = tensor(np.arange(1, len(data) + 1), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
    I = tensor(data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
    R = tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = tensor(data["cumulative_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
    return t, I, R, D

# Split and scale the data into training and validation sets
def split_and_scale_data(data, train_size, features, device):
    """Split and scale data into training and validation sets."""
    scaler = MinMaxScaler()
    scaler.fit(data[features])

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:]

    scaled_train_data = pd.DataFrame(scaler.transform(train_data[features]), columns=features)
    scaled_val_data = pd.DataFrame(scaler.transform(val_data[features]), columns=features)

    t_train, I_train, R_train, D_train = prepare_tensors(scaled_train_data, device)
    t_val, I_val, R_val, D_val = prepare_tensors(scaled_val_data, device)

    tensor_data = {
        "train": (t_train, I_train, R_train, D_train),
        "val": (t_val, I_val, R_val, D_val),
    }

    return tensor_data, scaler

# Example features and data split
features = ["active_cases", "recovered", "cumulative_deceased"]
train_size = 60  # days

tensor_data, scaler = split_and_scale_data(data, train_size, features, device)

# PINN loss function
def pinn_loss(tensor_data, parameters, model_output, t, N, sigma=1/5, beta=None, gamma=None, delta=None):
    """Physics-Informed Neural Network loss function."""
    S_pred, E_pred, I_pred, R_pred, D_pred = torch.split(model_output, 1, dim=1)
    
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
    
    # Data fitting loss for the predicted values vs. the true values
    data_fitting_loss = torch.mean((I_pred - tensor_data["train"][1]) ** 2) + torch.mean((R_pred - tensor_data["train"][2]) ** 2) + torch.mean((D_pred - tensor_data["train"][3]) ** 2)
    
    # Differential loss for the S, E, I, R, D compartments
    differential_loss = torch.mean((s_t - dSdt) ** 2) + torch.mean((e_t - dEdt) ** 2) + torch.mean((i_t - dIdt) ** 2) + torch.mean((r_t - dRdt) ** 2) + torch.mean((d_t - dDdt) ** 2)
    
    # Initial condition loss
    initial_condition_loss = torch.mean((S_pred[0] - 1) ** 2) + torch.mean(E_pred[0] ** 2) + torch.mean((I_pred[0] - tensor_data["train"][1][0]) ** 2) + torch.mean((R_pred[0] - tensor_data["train"][2][0]) ** 2) + torch.mean((D_pred[0] - tensor_data["train"][3][0]) ** 2)
    
    return data_fitting_loss + differential_loss + initial_condition_loss

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
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Initialize parameters and model
N = data["population"].values[0]
model = SEIRDNet(inverse=True, init_beta=0.3, init_gamma=0.1, init_delta=0.1, num_layers=10, hidden_neurons=32, retain_seed=100).to(device)

# Initialize optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=2000, gamma=0.9)

# Initialize early stopping
earlystopping = EarlyStopping(patience=100, verbose=False)

# Set the number of epochs for training
epochs = 50000

# Shuffle the data index
index = torch.randperm(len(tensor_data["train"][0]))

# List to store loss history
loss_history = []

# Training loop function
def train_loop(model, optimizer, scheduler, earlystopping, epochs, tensor_data, N, loss_history, index):
    """Training loop for the model."""
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        t, I, R, D = tensor_data["train"]
        t, I, R, D = t[index], I[index], R[index], D[index]
        model_output = model(t)
        loss = pinn_loss(tensor_data, model, model_output, t, N)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())
        earlystopping(loss.item())
        if earlystopping.early_stop:
            print("Early stopping")
            break
        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")
            
    return model, loss_history

# Train the model
model, loss_history = train_loop(model, optimizer, scheduler, earlystopping, epochs, tensor_data, N, loss_history, index)

# Plot the loss history
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss History")
plt.show()

# Switch the model to evaluation mode
model.eval()
with torch.no_grad():
    t_val, I_val, R_val, D_val = tensor_data["val"]
    model_output = model(t_val)
    S_pred, E_pred, I_pred, R_pred, D_pred = torch.split(model_output, 1, dim=1)

    # Inverse transform the scaled data
    S_pred = scaler.inverse_transform(S_pred.cpu().numpy())
    E_pred = scaler.inverse_transform(E_pred.cpu().numpy())
    I_pred = scaler.inverse_transform(I_pred.cpu().numpy())
    R_pred = scaler.inverse_transform(R_pred.cpu().numpy())
    D_pred = scaler.inverse_transform(D_pred.cpu().numpy())

    # Plot the predicted vs. true data
    plt.figure()
    plt.plot(data["active_cases"], label="True Active Cases", color="blue", linestyle="--")
    plt.plot(np.arange(train_size, len(data)), I_pred, label="Predicted Active Cases", color="red")
    plt.xlabel("Days")
    plt.ylabel("Active Cases")
    plt.title("Predicted vs. True Active Cases")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(data["recovered"], label="True Recovered", color="blue", linestyle="--")
    plt.plot(np.arange(train_size, len(data)), R_pred, label="Predicted Recovered", color="red")
    plt.xlabel("Days")
    plt.ylabel("Recovered")
    plt.title("Predicted vs. True Recovered")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(data["cumulative_deceased"], label="True Deceased", color="blue", linestyle="--")
    plt.plot(np.arange(train_size, len(data)), D_pred, label="Predicted Deceased", color="red")
    plt.xlabel("Days")
    plt.ylabel("Deceased")
    plt.title("Predicted vs. True Deceased")
    plt.legend()
    plt.show()

