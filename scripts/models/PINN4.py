import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm

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
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    df["S(t)"] = df["population"] - df["cumulative_confirmed"] - df["cumulative_deceased"] - df["recovered"]

    cols_to_smooth = ["S(t)", "cumulative_confirmed", "cumulative_deceased", "hospitalCases", "covidOccupiedMVBeds", "recovered", "active_cases", "new_deceased", "new_confirmed"]
    for col in cols_to_smooth:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    return df

# Load and preprocess the data
data = load_and_preprocess_data("../../data/hos_data/merged_data.csv", areaname="South West", recovery_period=21, rolling_window=7, start_date="2020-04-01", end_date="2020-12-31").drop(columns=["Unnamed: 0"], axis=1)

data.head(10)

def SEIRD_model(t, y, beta, gamma, mu, sigma, e, alpha, N):
    """SEIRD model differential equations."""
    S, E, I, R, D = y
    dSdt = -beta * S * (e * E + I) / N
    dEdt = beta * S * (e * E + I) / N - E / alpha
    dIdt = E / alpha - (gamma + mu) * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def prepare_tensors(data, device):
    """Prepare tensors for training."""
    t = tensor(range(1, len(data) + 1), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
    S = tensor(data["S(t)"].values, dtype=torch.float32).view(-1, 1).to(device)
    I = tensor(data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
    R = tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = tensor(data["new_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
    return t, S, I, R, D

def scale_data(data, features):
    """Scale the data using MinMaxScaler."""
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
    return scaled_data, scaler  

# Define features and data split
features = ["S(t)", "active_cases", "recovered", "new_deceased"]

# Scale the data
scaled_data, scaler = scale_data(data, features)

# Prepare tensors
t_data, S_data, I_data, R_data, D_data = prepare_tensors(scaled_data, device)

class ModifiedTanh(nn.Module):
    def __init__(self, alpha, epsilon):
        super(ModifiedTanh, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, x):
        return 0.5 * torch.tanh(self.alpha * x) + self.epsilon

class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        identity = x.clone()
        out = self.fc(x)
        out = self.activation(out)
        if out.shape == identity.shape:
            out = out + identity
        return out

class StateNN(nn.Module):
    """Epidemiological network for predicting SEIRD model outputs."""
    def __init__(self, num_layers=4, hidden_neurons=20):
        super(StateNN, self).__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(ResBlock(hidden_neurons, hidden_neurons))
        layers.append(nn.Linear(hidden_neurons, 5))  # Adjust the output size to 5 (S, E, I, R, D)
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)

class ParamNN(nn.Module):
    """Neural network for predicting time-varying parameters."""
    def __init__(self, num_layers=4, hidden_neurons=20):
        super(ParamNN, self).__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(ResBlock(hidden_neurons, hidden_neurons))
        layers.append(nn.Linear(hidden_neurons, 3))  # Adjust the output size to 3 (beta, gamma, mu)
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        params = self.net(t)
        # Ensure beta, gamma, and mu are in a valid range
        beta = torch.sigmoid(params[:, 0]) * 0.5  # range: [0, 0.5]
        gamma = torch.sigmoid(params[:, 1]) * 0.1  # range: [0, 0.1]
        mu = torch.sigmoid(params[:, 2]) * 0.1  # range: [0, 0.1]
        return beta, gamma, mu

def pinn_loss(t, data, state_nn, param_nn, N, sigma, alpha, epsilon):
    """Physics-Informed Neural Network loss function."""
    
    # Predicted states
    states_pred = state_nn(t)
    S_pred, E_pred, I_pred, R_pred, D_pred = states_pred[:, 0], states_pred[:, 1], states_pred[:, 2], states_pred[:, 3], states_pred[:, 4]
    
    # Compute gradients
    S_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    E_t = grad(E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True)[0]   
    I_t = grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    R_t = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    D_t = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]
    
    # Predicted parameters
    beta_pred, gamma_pred, mu_pred = param_nn(t)
    
    # SEIRD model residuals
    e_tensor = torch.tensor(epsilon, dtype=torch.float32, device=device, requires_grad=True)
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device, requires_grad=True)
    
    e = torch.tanh(e_tensor)
    alpha = 2 * torch.tanh(alpha_tensor)
    
    dSdt, dEdt, dIdt, dRdt, dDdt = SEIRD_model(t, [S_pred, E_pred, I_pred, R_pred, D_pred], beta_pred, gamma_pred, mu_pred, sigma, e, alpha, N)
    
    # Compute data loss (MSE_u)
    S_data, I_data, R_data, D_data = data
    loss_data = torch.mean((S_pred - S_data)**2) + torch.mean((I_pred - I_data)**2) + torch.mean((R_pred - R_data)**2) + torch.mean((D_pred - D_data)**2)
    
    # Compute physics loss (MSE_f)
    loss_physics = torch.mean((S_t - dSdt)**2) + torch.mean((E_t - dEdt)**2) + torch.mean((I_t - dIdt)**2) + torch.mean((R_t - dRdt)**2) + torch.mean((D_t - dDdt)**2)
    
    # initial condition loss
    loss_initial = torch.mean((S_pred[0] - S_data[0])**2) + torch.mean((I_pred[0] - I_data[0])**2) + torch.mean((R_pred[0] - R_data[0])**2) + torch.mean((D_pred[0] - D_data[0])**2)
    
    # Total loss    
    total_loss = loss_data + loss_physics + loss_initial
    
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
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Hyperparameters
learning_rate = 1e-4
num_epochs = 50000
sigma = 1/5
N = data["population"].values[0]
alpha = 0.5
epsilon = 0.1

# Instantiate the neural networks with custom activation function
state_nn = StateNN(num_layers=6, hidden_neurons=32).to(device)
param_nn = ParamNN(num_layers=6, hidden_neurons=32).to(device)

# Optimizers
optimizer_state = optim.Adam(state_nn.parameters(), lr=learning_rate)
optimizer_param = optim.Adam(param_nn.parameters(), lr=learning_rate)

# Early stopping criteria
early_stopping = EarlyStopping(patience=20, verbose=False)

# Training loop
loss_history = []
for epoch in tqdm(range(num_epochs)):
    state_nn.train()
    param_nn.train()
    
    optimizer_state.zero_grad()
    optimizer_param.zero_grad()
    
    # Prepare time tensor
    # Prepare time tensor
    t = t_data
    
    data_tensors = (S_data, I_data, R_data, D_data)
    
    # Compute loss
    loss = pinn_loss(t, data_tensors, state_nn, param_nn, N, sigma, alpha, epsilon)
    
    # Backpropagation
    loss.backward()
    
    optimizer_state.step()
    optimizer_param.step()
    
    loss_history.append(loss.item())
    
    if epoch % 500 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")
    
    if early_stopping(loss.item()):
        print(f"Early stopping at epoch {epoch}. No improvement in loss for {early_stopping.patience} epochs.")
        break

# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(np.log10(loss_history), label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predict and plot the results
state_nn.eval()
param_nn.eval()

with torch.no_grad():
    t = torch.linspace(0, 1, steps=len(t_data)).view(-1, 1).to(device)
    states_pred = state_nn(t).cpu().numpy()
    S_pred, E_pred, I_pred, R_pred, D_pred = states_pred[:, 0], states_pred[:, 1], states_pred[:, 2], states_pred[:, 3], states_pred[:, 4]

    # Inverse transform to get back to original scale for each variable separately
    I_pred = scaler.inverse_transform(np.concatenate([I_pred.reshape(-1, 1)] * len(features), axis=1))[:, 0]
    R_pred = scaler.inverse_transform(np.concatenate([R_pred.reshape(-1, 1)] * len(features), axis=1))[:, 0]
    D_pred = scaler.inverse_transform(np.concatenate([D_pred.reshape(-1, 1)] * len(features), axis=1))[:, 0]

# Plot infected predicted vs real data
plt.figure(figsize=(10, 5))
plt.plot(data["date"], I_pred, label='Infected (Predicted)')
plt.plot(data["date"], data["active_cases"], label='Infected (Actual)', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Infected Population')
plt.legend()
plt.show()

# Plot recovered predicted vs real data
plt.figure(figsize=(10, 5))
plt.plot(data["date"], R_pred, label='Recovered (Predicted)')
plt.plot(data["date"], data["recovered"], label='Recovered (Actual)', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Recovered Population')
plt.legend()
plt.show()

# Plot deceased predicted vs real data
plt.figure(figsize=(10, 5))
plt.plot(data["date"], D_pred, label='Deceased (Predicted)')
plt.plot(data["date"], data["new_deceased"], label='Deceased (Actual)', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Deceased Population')
plt.legend()
plt.show()
