import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from scipy.integrate import odeint

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

# data = pd.read_csv("../../data/hos_data/merged_data.csv").drop(columns=["Unnamed: 0"], axis=1)

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
    df = df[df["nhs_region"] == areaname].reset_index(drop=True)
    df = df[::-1].reset_index(drop=True)  # Reverse dataset if needed

    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df["cumulative_deceased"].shift(recovery_period)
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)
    df["active_cases"] = df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]
    df["S(t)"] = df["population"] - df["active_cases"] - df["recovered"] - df["cumulative_deceased"]

    cols_to_smooth = ["S(t)", "cumulative_confirmed", "cumulative_deceased", "hospitalCases", "covidOccupiedMVBeds", "recovered", "active_cases"]
    for col in cols_to_smooth:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    return df

# Load and preprocess the data
data = load_and_preprocess_data("../../data/hos_data/merged_data.csv", areaname="South West", recovery_period=21, start_date="2020-04-01", end_date="2021-12-31")

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
            self._beta = nn.Parameter(torch.tensor([init_beta if init_beta is not None else torch.rand(1)], device='cpu'), requires_grad=True)
            self._gamma = nn.Parameter(torch.tensor([init_gamma if init_gamma is not None else torch.rand(1)], device='cpu'), requires_grad=True)
            self._delta = nn.Parameter(torch.tensor([init_delta if init_delta is not None else torch.rand(1)], device='cpu'), requires_grad=True)
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
    t = tensor(range(1, len(data) + 1), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
    S = tensor(data["S(t)"].values, dtype=torch.float32).view(-1, 1).to(device)
    I = tensor(data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
    R = tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = tensor(data["new_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
    return t, S, I, R, D

# Split and scale the data into training and validation sets
def split_and_scale_data(data, train_size, features, device):
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
features = ["S(t)", "active_cases", "recovered", "new_deceased"]
train_size = 60  # days

# Assuming 'data' is a DataFrame containing the relevant columns
tensor_data, scaler = split_and_scale_data(data, train_size, features, device)

# PINN loss function
def pinn_loss(tensor_data, parameters, model_output, t, N, device):
    """Compute the PINN loss."""
    t_train, S_train, I_train, R_train, D_train = tensor_data["train"]
    t_val, S_val, I_val, R_val, D_val = tensor_data["val"]

    S = torch.cat([S_train, S_val], dim=0)
    I = torch.cat([I_train, I_val], dim=0)
    R = torch.cat([R_train, R_val], dim=0)
    D = torch.cat([D_train, D_val], dim=0)

    beta_pred = parameters[:, 0].squeeze()
    gamma_pred = parameters[:, 1].squeeze()
    delta_pred = parameters[:, 2].squeeze()
    sigma = 1/5  # Fixed sigma value

    S_pred, E_pred, I_pred, R_pred, D_pred = model_output.unbind(1)

    s_t = grad(outputs=S_pred, inputs=t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    i_t = grad(outputs=I_pred, inputs=t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    r_t = grad(outputs=R_pred, inputs=t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    d_t = grad(outputs=D_pred, inputs=t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]

    # Calculate the model derivatives based on the SEIRD model
    dSdt_pred = -beta_pred * S_pred * I_pred / N
    dIdt_pred = sigma * E_pred - (gamma_pred + delta_pred) * I_pred
    dRdt_pred = gamma_pred * I_pred
    dDdt_pred = delta_pred * I_pred

    # Calculate losses
    data_loss = torch.mean((S - S_pred) ** 2 + (I - I_pred) ** 2 + (R - R_pred) ** 2 + (D - D_pred) ** 2)
    physics_loss = torch.mean((s_t - dSdt_pred) ** 2 + (i_t - dIdt_pred) ** 2 + (r_t - dRdt_pred) ** 2 + (d_t - dDdt_pred) ** 2)
    initial_condition_loss = torch.mean((S[0] - S_pred[0]) ** 2 + (I[0] - I_pred[0]) ** 2 + (R[0] - R_pred[0]) ** 2 + (D[0] - D_pred[0]) ** 2)
    boundary_condition_loss = torch.mean((S[-1] - S_pred[-1]) ** 2 + (I[-1] - I_pred[-1]) ** 2 + (R[-1] - R_pred[-1]) ** 2 + (D[-1] - D_pred[-1]) ** 2)
    reg_loss = torch.mean(beta_pred**2 + gamma_pred**2 + delta_pred**2)

    # Total loss
    loss = data_loss + physics_loss + initial_condition_loss + boundary_condition_loss + reg_loss

    return loss

class EarlyStopping:
    """Early stopping utility to stop training when validation loss doesn't improve."""
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train_model(tensor_data, model, beta_net, model_optimizer, params_optimizer, model_scheduler, params_scheduler, N, early_stopping, num_epochs=5000):
    """Train the model."""
    loss_history = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        beta_net.train()

        running_loss = 0.0

        train_tensors, val_tensors = tensor_data["train"], tensor_data["val"]
        t_train, S_train, I_train, R_train, D_train = train_tensors

        model_optimizer.zero_grad()
        params_optimizer.zero_grad()

        params = beta_net(t_train)
        model_output = model(t_train)

        loss = pinn_loss(tensor_data, params, model_output, t_train, N, device)
        running_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(beta_net.parameters(), max_norm=1.0)
        model_optimizer.step()
        params_optimizer.step()

        model_scheduler.step()
        params_scheduler.step()

        loss_history.append(running_loss)

        if early_stopping(running_loss):
            print(f"Early stopping at epoch {epoch}. No improvement in loss for {early_stopping.patience} epochs.")
            break

        if epoch % 500 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.6f}")  # Print every 500 epochs

    print("Finished Training")
    return loss_history

# Initialize the models
model = SEIRDNet(num_layers=10, hidden_neurons=20).to(device)
beta_net = SEIRDNet(inverse=True, init_beta=0.1, init_gamma=0.01, init_delta=0.01, num_layers=6, hidden_neurons=32).to(device)

# Initialize the optimizers
model_optimizer = optim.Adam(model.parameters(), lr=1e-3)
params_optimizer = optim.Adam(beta_net.parameters(), lr=1e-3)

# Define the learning rate scheduler
model_scheduler = StepLR(model_optimizer, step_size=5000, gamma=0.998)
params_scheduler = StepLR(params_optimizer, step_size=5000, gamma=0.998)

# Early stopping criteria
early_stopping = EarlyStopping(patience=20, verbose=False)

# Total population  
N = data["population"].values[0]

# Train the model
loss_history = train_model(tensor_data, model, beta_net, model_optimizer, params_optimizer, model_scheduler, params_scheduler, N, early_stopping, num_epochs=50000)

# Plot training loss in log scale
plt.figure(figsize=(10, 5))
plt.plot(np.log10(loss_history), label="Training Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.title("Training Loss over Epochs (Log Scale)")
plt.legend()
plt.show()

# Initial conditions for the SEIRD model based on the data
S0 = data["S(t)"].values[0]
I0 = data["active_cases"].values[0]
R0 = data["recovered"].values[0]
D0 = data["new_deceased"].values[0]
E0 = 0  # Initial exposed cases assumed to be zero

# Simulation time points
t = np.linspace(0, train_size, train_size + 1)[:-1]

u0 = [S0, E0, I0, R0, D0]  # initial conditions vector

# Extract the parameters from the trained model
params = beta_net(tensor(t, dtype=torch.float32).view(-1, 1).to(device)).detach().cpu().numpy()

beta = params[:, 0]
gamma = params[:, 1]
delta = params[:, 2]
sigma = 1 / 5  # Fixed sigma value

def integrate_step_by_step(u0, t, params, N):
    """Integrate the ODE step by step to update initial conditions at each step."""
    res = []
    u = u0
    for i in range(len(t) - 1):
        t_span = [t[i], t[i + 1]]
        beta, gamma, delta = params[i]
        sol = odeint(SEIRD_model, u, t_span, args=(beta, sigma, gamma, delta, N))
        u = sol[-1]  # Update the initial condition for the next step
        res.append(u)
    return np.array(res)

# Integrate the SEIRD equations over the time grid, t.
res = integrate_step_by_step(u0, t, params, N)
S_ode, E_ode, I_ode, R_ode, D_ode = res.T

# Plot the results versus the original data
plt.figure(figsize=(16, 9))
plt.plot(t, data["active_cases"].values[:train_size], label="I(t) (Data)", color="red")
plt.plot(t, I_ode, label="I(t) (ODE)", linestyle="--", color="red")
plt.xlabel("Time (days)")
plt.ylabel("Number of Active Cases")
plt.title("Active Cases (I(t))")
plt.tight_layout()
plt.legend()
plt.show()

def plot_results_comparison(country, data_type, real_data, pre_data, ode_data, train_size, plot_type, path_results):
    """Plot the comparison of real data, predicted data and ODE model data."""
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    plt.figure(figsize=(16, 9))
    t = np.linspace(0, len(pre_data), len(pre_data) + 1)[:-1]

    plt.plot(t, real_data, color='black', label=f'{data_type}_real')
    plt.scatter(t[:train_size], real_data[:train_size], color='black', marker='*', label=f'{data_type}_train')
    plt.plot(t, pre_data, color='red', label=f'{data_type}_pinn')
    plt.plot(t, ode_data, color='green', label=f'{data_type}_ode')
    
    plt.xlabel('Days')
    plt.ylabel(f'{data_type}')
    plt.title(f'{country} {data_type} Results')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(path_results, f'{country}_{data_type}_results_{plot_type}_comparison.pdf'), dpi=600)
    plt.close()

# Example usage of plot_results_comparison
plot_results_comparison(
    country="South West",
    data_type="Active Cases",
    real_data=data["active_cases"].values,
    pre_data=I_ode,
    ode_data=I_ode,
    train_size=train_size,
    plot_type="SEIRD",
    path_results="./results"
)
