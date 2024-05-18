import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from collections import deque
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tqdm import tqdm

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

    cols_to_smooth = ["cumulative_confirmed", "cumulative_deceased", "hospitalCases", "covidOccupiedMVBeds", "recovered", "active_cases", "new_deceased", "new_confirmed"]
    for col in cols_to_smooth:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    return df

# Load and preprocess the data
data = load_and_preprocess_data("../../data/hos_data/merged_data.csv", areaname="South West", recovery_period=21, start_date="2020-04-01", end_date="2020-12-31").drop(columns=["Unnamed: 0"], axis=1)

days = len(data)

class SEIRDNet(nn.Module):
    """Epidemiological network for predicting SEIRD model outputs."""
    def __init__(self, num_layers=4, hidden_neurons=20):
        super(SEIRDNet, self).__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 5))  # Adjust the output size to 5 (S, E, I, R, D)
        self.net = nn.Sequential(*layers)
        self.init_xavier()

    def forward(self, t):
        return self.net(t)
    
    def init_xavier(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(init_weights)

class ParameterNet(nn.Module):
    """Network for estimating SEIRD model parameters."""
    def __init__(self, num_layers=4, hidden_neurons=20):
        super(ParameterNet, self).__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 3))  # Adjust the output size to 3 (beta, gamma, delta)
        self.net = nn.Sequential(*layers)
        self.init_xavier()

    def forward(self, t):
        params = self.net(t)
        beta = torch.sigmoid(params[:, 0]) * 0.9 + 0.1
        gamma = torch.sigmoid(params[:, 1]) * 0.09 + 0.01
        delta = torch.sigmoid(params[:, 2]) * 0.09 + 0.01
        return beta, gamma, delta
    
    def init_xavier(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(init_weights)

def SEIRD_model(t, y, beta, gamma, delta, sigma, N):
    """SEIRD model differential equations."""
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (gamma + delta) * I
    dRdt = gamma * I
    dDdt = delta * I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def prepare_tensors(data, device):
    """Prepare tensors for training."""
    t = tensor(range(1, len(data) + 1), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
    I = tensor(data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
    R = tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = tensor(data["new_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
    return t, I, R, D

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

# Define features and data split
features = ["active_cases", "recovered", "new_deceased"]
train_size = 60  # days

# Split and scale data
tensor_data, scaler = split_and_scale_data(data, train_size, features, device)

# PINN loss function
def pinn_loss(tensor_data, parameters, model_output, t, N, sigma=1/5):
    """Physics-Informed Neural Network loss function."""
    mse_loss = nn.MSELoss()

    S_pred, E_pred, I_pred, R_pred, D_pred = model_output[:, 0], model_output[:, 1], model_output[:, 2], model_output[:, 3], model_output[:, 4]
    
    t_train, I_train, R_train, D_train = tensor_data["train"]
    t_val, I_val, R_val, D_val = tensor_data["val"]
    
    # Calculate S_train and S_val
    S_train = N - I_train - R_train - D_train
    S_val = N - I_val - R_val - D_val
    
    # Placeholder for E_train and E_val, since they were not provided in the dataset
    E_train = torch.zeros_like(I_train)
    E_val = torch.zeros_like(I_val)
    
    # Concatenate training and validation data to form the entire dataset
    S = torch.cat([S_train, S_val], dim=0)
    E = torch.cat([E_train, E_val], dim=0)
    I = torch.cat([I_train, I_val], dim=0)
    R = torch.cat([R_train, R_val], dim=0)
    D = torch.cat([D_train, D_val], dim=0)

    s_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    e_t = grad(E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True)[0]
    i_t = grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    r_t = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    d_t = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]
    
    beta, gamma, delta = parameters
    
    dSdt, dEdt, dIdt, dRdt, dDdt = SEIRD_model(
        t, [S_pred, E_pred, I_pred, R_pred, D_pred],
        beta, gamma, delta, sigma, N
    )
    
    # Data loss using the train_size to forecast the entire dataset
    data_loss = torch.mean((I_pred[:train_size] - I_train) ** 2) + torch.mean((R_pred[:train_size] - R_train) ** 2) + torch.mean((D_pred[:train_size] - D_train) ** 2)
    
    # Physics loss
    physics_loss = torch.mean((s_t - dSdt) ** 2) + torch.mean((e_t - dEdt) ** 2) + torch.mean((i_t - dIdt) ** 2) + torch.mean((r_t - dRdt) ** 2) + torch.mean((d_t - dDdt) ** 2)
    
    # Initial condition loss
    initial_condition_loss = torch.mean((S_pred[0] - S[0]) ** 2) + torch.mean((E_pred[0] - E[0]) ** 2) + torch.mean((I_pred[0] - I[0]) ** 2) + torch.mean((R_pred[0] - R[0]) ** 2) + torch.mean((D_pred[0] - D[0]) ** 2)
    
    # Boundary condition loss
    boundary_condition_loss = torch.mean((S_pred[-1] - S[-1]) ** 2) + torch.mean((E_pred[-1] - E[-1]) ** 2) + torch.mean((I_pred[-1] - I[-1]) ** 2) + torch.mean((R_pred[-1] - R[-1]) ** 2) + torch.mean((D_pred[-1] - D[-1]) ** 2)
    
    total_loss = data_loss + physics_loss + initial_condition_loss + boundary_condition_loss
    
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

# Initialize the models
model = SEIRDNet(num_layers=6, hidden_neurons=32).to(device)
param_net = ParameterNet(num_layers=5, hidden_neurons=32).to(device)

# Initialize the optimizers
model_optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
param_optimizer = optim.Adam(param_net.parameters(), lr=1e-3, weight_decay=1e-5)

# Learning rate scheduler
scheduler = StepLR(model_optimizer, step_size=2000, gamma=0.9)

# Early stopping criteria
early_stopping = EarlyStopping(patience=20, verbose=False)

# Loss history
loss_history = []

# Total population
N = data["population"].values[0]  # Assuming the population is constant and given in the data

def train_model(tensor_data, model, param_net, model_optimizer, param_optimizer, scheduler, early_stopping, num_epochs=1000):
    """Training loop for the model."""
    for epoch in tqdm(range(num_epochs)):
        model.train()
        param_net.train()

        model_optimizer.zero_grad()
        param_optimizer.zero_grad()

        t_train, I_train, R_train, D_train = tensor_data["train"]
        
        # Shuffle the training indices
        idx = torch.randperm(t_train.size(0))
        t_train_shuffled = t_train[idx]

        params = param_net(t_train_shuffled)
        model_output = model(t_train_shuffled)

        loss = pinn_loss(tensor_data, params, model_output, t_train_shuffled, N)
        loss.backward()
        
        model_optimizer.step()
        param_optimizer.step()
        
        scheduler.step()

        loss_history.append(loss.item())

        if early_stopping(loss.item()):
            print(f"Early stopping at epoch {epoch}. No improvement in loss for {early_stopping.patience} epochs.")
            break

        if epoch % 500 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}: Loss: {loss.item():.6f}")

    print("Finished Training")
    return loss_history, model, param_net

# Train the model
loss_history, model, param_net = train_model(tensor_data, model, param_net, model_optimizer, param_optimizer, scheduler, early_stopping, num_epochs=50000)

# Plot training loss in log scale
plt.figure(figsize=(10, 5))
plt.plot(np.log10(loss_history), label="Training Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.title("Training Loss over Epochs (Log Scale)")
plt.legend()
plt.show()

# Total time frame from the start of the training data to the end of the validation data
t_total = np.linspace(1, days, days + 1)[:-1]

model.eval()
param_net.eval()

# Predict the SEIRD model outputs
params = param_net(tensor(t_total, dtype=torch.float32).view(-1, 1).to(device))
model_output = model(tensor(t_total, dtype=torch.float32).view(-1, 1).to(device))
