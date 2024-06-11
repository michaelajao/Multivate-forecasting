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
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Ensure the folders exist
os.makedirs("../../models", exist_ok=True)
os.makedirs("../../reports/figures", exist_ok=True)
os.makedirs("../../reports/results", exist_ok=True)
os.makedirs("../../reports/parameters", exist_ok=True)

# Set CUDA_LAUNCH_BLOCKING for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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

# Set matplotlib style and parameters
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.size": 14,
        "figure.figsize": [10, 6],
        "text.usetex": False,
        "figure.facecolor": "white",
        "figure.autolayout": True,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
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
        "legend.fontsize": 12,
        "legend.frameon": False,
        "legend.loc": "best",
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "xtick.labelsize": 12,
        "xtick.direction": "in",
        "xtick.top": False,
        "ytick.labelsize": 12,
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

def normalized_root_mean_square_error(y_true, y_pred):
    """Calculate the Normalized Root Mean Square Error (NRMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))

def safe_mean_absolute_scaled_error(y_true, y_pred, y_train, epsilon=1e-10):
    """Calculate the Mean Absolute Scaled Error (MASE) safely."""
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    d = max(d, epsilon)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

def calculate_errors(y_true, y_pred, y_train, train_size, areaname):
    """Calculate and print various error metrics."""
    mape = mean_absolute_percentage_error(y_true, y_pred)
    nrmse = normalized_root_mean_square_error(y_true, y_pred)
    mase = safe_mean_absolute_scaled_error(y_true, y_pred, y_train)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Normalized Root Mean Square Error (NRMSE): {nrmse:.4f}")
    print(f"Mean Absolute Scaled Error (MASE): {mase:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    
    return mape, nrmse, mase, rmse, mae, mse

def calculate_all_metrics(actual, predicted, train_data, label, train_size, areaname):
    """Calculate metrics for each state."""
    print(f"\nMetrics for {label}:")
    mape, nrmse, mase, rmse, mae, mse = calculate_errors(actual, predicted, train_data, train_size, areaname)
    return mape, nrmse, mase, rmse, mae, mse

# Define the SEIRD model differential equations
def seird_model(y, t, N, beta, alpha, rho, ds, da, omega, dH, mu, gamma_c, delta_c, eta):
    S, E, Is, Ia, H, C, R, D = y
    
    dSdt = -beta * (Is + Ia) / N * S + eta * R
    dEdt = beta * (Is + Ia) / N * S - alpha * E
    dIsdt = alpha * rho * E - ds * Is
    dIadt = alpha * (1 - rho) * E - da * Ia
    dHdt = ds * omega * Is - dH * H - mu * H
    dCdt = dH * (1 - omega) * H - gamma_c * C - delta_c * C
    dRdt = ds * (1 - omega) * Is + da * Ia + dH * (1 - mu) * H + gamma_c * C - eta * R
    dDdt = mu * H + delta_c * C
    
    return [dSdt, dEdt, dIsdt, dIadt, dHdt, dCdt, dRdt, dDdt]


areaname = "East of England"
def load_preprocess_data(filepath, areaname, recovery_period=16, rolling_window=7, end_date=None):
    """Load and preprocess the COVID-19 data."""
    df = pd.read_csv(filepath)
    
    # Select the columns of interest
    df = df[df["areaName"] == areaname].reset_index(drop=True)
    
    # # reset the index
    # df = df[::-1].reset_index(drop=True)  # Reverse dataset if needed
    
    # Convert the date column to datetime
    df["date"] = pd.to_datetime(df["date"])
    
    
    # calculate the recovered column from data
    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df["cumulative_deceased"].shift(recovery_period)
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)
    
    # select data up to the end_date
    if end_date:
        df = df[df["date"] <= end_date]
        
    # calculate the susceptible column from data
    df["susceptible"] = df["population"] - df["cumulative_confirmed"] - df["cumulative_deceased"] - df["recovered"]
    
    cols_to_smooth = ["susceptible", "cumulative_confirmed", "cumulative_deceased", "hospitalCases", "covidOccupiedMVBeds", "new_deceased", "new_confirmed", "newAdmissions", "cumAdmissions", "recovered"]
    for col in cols_to_smooth:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)
        
    # normalize the data with the population
    # df["cumulative_confirmed"] = df["cumulative_confirmed"] / df["population"]
    # df["cumulative_deceased"] = df["cumulative_deceased"] / df["population"]
    # df["hospitalCases"] = df["hospitalCases"] / df["population"]
    # df["covidOccupiedMVBeds"] = df["covidOccupiedMVBeds"] / df["population"]
    # df["new_deceased"] = df["new_deceased"] / df["population"]
    # df["new_confirmed"] = df["new_confirmed"] / df["population"]
    # df["recovered"] = df["recovered"] / df["population"]
    # df["population"] = df["population"] / df["population"]
    # df["susceptible"] = df["susceptible"] / df["population"]
    # df["newAdmissions"] = df["newAdmissions"] / df["population"]
    # df["cumAdmissions"] = df["cumAdmissions"] / df["population"]
    
    return df

data = load_preprocess_data("../../data/processed/merged_nhs_covid_data.csv", areaname, recovery_period=21, rolling_window=7, end_date="2020-08-31")

plt.plot(data["date"], data["new_deceased"])
plt.title("New Deceased over time")
plt.xlabel("Date")
plt.ylabel("New Deceased")
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig("../../reports/figures/S(t)_over_time.pdf")
plt.show()

def prepare_tensors(data, device):
    """Prepare tensors for training."""
    I = tensor(data["new_confirmed"].values, dtype=torch.float32).view(-1, 1).to(device)
    H = tensor(data["newAdmissions"].values, dtype=torch.float32).view(-1, 1).to(device)
    C = tensor(data["covidOccupiedMVBeds"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = tensor(data["new_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
    R = tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    return I, H, C, D, R


def split_and_scale_data(data, train_size, features, device):
    """Split and scale data into training and validation sets."""
    scaler = MinMaxScaler()
    scaler.fit(data[features])

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:]

    scaled_train_data = pd.DataFrame(scaler.transform(train_data[features]), columns=features)
    scaled_val_data = pd.DataFrame(scaler.transform(val_data[features]), columns=features)

    Is_train, H_train, C_train, D_train, R_train = prepare_tensors(scaled_train_data, device)
    Is_val, H_val, C_val, D_val, R_val = prepare_tensors(scaled_val_data, device)

    tensor_data = {
        "train": (Is_train, H_train, C_train, D_train, R_train),
        "val": (Is_val, H_val, C_val, D_val, R_val),
    }
    
    return tensor_data, scaler


features = ["new_confirmed", "newAdmissions", "covidOccupiedMVBeds", "new_deceased", "recovered"]

# set the train size in days
train_size = 80

tensor_data, scaler = split_and_scale_data(data, train_size, features, device)

# Define the neural network model
class StateNN(nn.Module):
    def __init__(self, inverse=False, init_beta=None, init_omega=None, init_mu=None, init_gamma=None, init_delta=None, init_eta=None, retrain_seed=42, num_layers=4, hidden_neurons=20):
        super(StateNN, self).__init__()
        self.retrain_seed = retrain_seed
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 8))
        self.net = nn.Sequential(*layers)

        if inverse:
            self._beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32).to(device), requires_grad=True)
            self._omega = nn.Parameter(torch.tensor(init_omega, dtype=torch.float32).to(device), requires_grad=True)
            self._mu = nn.Parameter(torch.tensor(init_mu, dtype=torch.float32).to(device), requires_grad=True)
            self._gamma_c = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32).to(device), requires_grad=True)
            self._delta_c = nn.Parameter(torch.tensor(init_delta, dtype=torch.float32).to(device), requires_grad=True)
            self._eta = nn.Parameter(torch.tensor(init_eta, dtype=torch.float32).to(device), requires_grad=True)
        else:
            self._beta = None
            self._omega = None
            self._mu = None
            self._gamma_c = None
            self._delta_c = None
            self._eta = None

        self.init_xavier()

    def forward(self, t):
        return self.net(t)

    @property
    def beta(self):
        return torch.sigmoid(self._beta) *  0.9 + 0.1

    @property
    def omega(self):
        return torch.sigmoid(self._omega) * 0.09 + 0.01

    @property
    def delta(self):
        return torch.sigmoid(self._mu) * 0.09 + 0.01
    
    @property
    def gamma_c(self):
        return torch.sigmoid(self._gamma_c) * 0.09 + 0.01
    
    @property
    def delta_c(self):
        return torch.sigmoid(self._delta_c) * 0.09 + 0.01
    
    @property
    def eta(self):
        return torch.sigmoid(self._eta) * 0.09 + 0.01
    

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        self.apply(init_weights)


def einn_loss(model_output, tensor_data, parameters, t, train_size, model, lambda_reg=1e-4):
    """Compute the loss function for the EINN model with L2 regularization."""
    
    # Split the model output into the different compartments
    S_pred, E_pred, Ia_pred, Is_pred, H_pred, C_pred, R_pred, D_pred = torch.split(model_output, 1, dim=1)
    
    # Extract training and validation data from tensor_data
    Is_train, H_train, C_train, D_train, R_train = tensor_data["train"]
    Is_val, H_val, C_val, D_val, R_val = tensor_data["val"]
    
    # Normalize the data
    N = 1
    
    # Combine training and validation data for total data
    Is_total = torch.cat([Is_train, Is_val])
    H_total = torch.cat([H_train, H_val])
    C_total = torch.cat([C_train, C_val])
    D_total = torch.cat([D_train, D_val])
    R_total = torch.cat([R_train, R_val])

    
    # Compute the total number of exposed and infectious individuals
    E_total = torch.zeros_like(Is_total)  # Initialize as a tensor of zeros
    Ia_total = torch.zeros_like(Is_total)  # Initialize as a tensor of zeros
    S_total = N - E_total - Ia_total - Is_total - H_total - C_total - R_total - D_total
    
    # Constants based on the table provided
    rho = 0.80  # Proportion of symptomatic infections
    alpha = 1 / 5  # Incubation period (5 days)
    d_s = 1 / 4 # Infectious period for symptomatic (4 days)
    d_a = 1 / 7  # Infectious period for asymptomatic (7 days)
    d_h = 1 / 13.4  # Hospitalization days (13.4 days)
    
    # learned parameters
    beta = parameters.beta
    omega = parameters.omega
    mu = parameters.delta
    gamma_c = parameters.gamma_c
    delta_c = parameters.delta_c
    eta = parameters.eta
    
    # Compute the loss function
    S_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    E_t = grad(E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True)[0]
    Ia_t = grad(Ia_pred, t, grad_outputs=torch.ones_like(Ia_pred), create_graph=True)[0]
    Is_t = grad(Is_pred, t, grad_outputs=torch.ones_like(Is_pred), create_graph=True)[0]
    H_t = grad(H_pred, t, grad_outputs=torch.ones_like(H_pred), create_graph=True)[0]
    C_t = grad(C_pred, t, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
    R_t = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    D_t = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]
    
    # Compute the differential equations
    dSdt = -beta * (Is_total + Ia_total) / N * S_total + eta * R_total
    dEdt = beta * (Is_total + Ia_total) / N * S_total - alpha * E_total
    dIsdt = alpha * rho * E_total - d_s * Is_total
    dIadt = alpha * (1 - rho) * E_total - d_a * Ia_total
    dHdt = d_s * omega * Is_total - d_h * H_total - mu * H_total
    dCdt = d_h * (1 - omega) * H_total - gamma_c * C_total - delta_c * C_total
    dRdt = d_s * (1 - omega) * Is_total + d_a * Ia_total + d_h * (1 - mu) * H_total + gamma_c * C_total - eta * R_total
    dDdt = mu * H_total + delta_c * C_total
    
    # Randomly shuffle the indices
    if train_size is not None:
        index = torch.randperm(train_size)
    else:
        index = torch.arange(len(t))
        
    # Compute the loss function
    # data loss
    data_loss = (
        torch.mean((S_pred[index] - S_total[index]) ** 2)
        + torch.mean((E_pred[index] - E_total[index]) ** 2)
        + torch.mean((Is_pred[index] - Is_total[index]) ** 2)
        + torch.mean((Ia_pred[index] - Ia_total[index]) ** 2)
        + torch.mean((H_pred[index] - H_total[index]) ** 2)
        + torch.mean((C_pred[index] - C_total[index]) ** 2)
        + torch.mean((D_pred[index] - D_total[index]) ** 2)
        + torch.mean((R_pred[index] - R_total[index]) ** 2)
    )
    
    # residual loss
    residual_loss = (
        torch.mean((S_t - dSdt) ** 2)
        + torch.mean((E_t - dEdt) ** 2)
        + torch.mean((Ia_t - dIadt) ** 2)
        + torch.mean((Is_t - dIsdt) ** 2)
        + torch.mean((H_t - dHdt) ** 2)
        + torch.mean((C_t - dCdt) ** 2)
        + torch.mean((R_t - dRdt) ** 2)
        + torch.mean((D_t - dDdt) ** 2)
    )
    
    # Initial condition loss
    S0, E0, Ia0, Is0, H0, C0, R0, D0 = S_total[0], E_total[0], Ia_total[0], Is_total[0], H_total[0], C_total[0], R_total[0], D_total[0]
    initial_loss = (
        (S_pred[0] - S0) ** 2
        + (E_pred[0] - E0) ** 2
        + (Ia_pred[0] - Ia0) ** 2
        + (Is_pred[0] - Is0) ** 2
        + (H_pred[0] - H0) ** 2
        + (C_pred[0] - C0) ** 2
        + (R_pred[0] - R0) ** 2
        + (D_pred[0] - D0) ** 2
    )
    
    # L2 regularization term
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param)

    # total loss
    loss = data_loss + residual_loss + initial_loss + lambda_reg * l2_reg
    
    return loss

# early stopping
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0
        self.loss_history = deque(maxlen=patience + 1)
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.best_model = model.state_dict()


model = StateNN(
    inverse=True,
    init_beta=0.1,
    init_omega=0.01,
    init_mu=0.01,
    init_gamma=0.01,
    init_delta=0.01,
    init_eta=0.01,
    retrain_seed=seed,
    num_layers=8,
    hidden_neurons=32
).to(device)
            
N = data["population"].iloc[0]
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = StepLR(optimizer, step_size=10000, gamma=0.9)
earlystopping = EarlyStopping(patience=100, verbose=False)
num_epochs = 100000

t = torch.tensor(np.arange(1, len(data) + 1), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)

index = torch.randperm(len(tensor_data["train"][0]))

loss_history = []

# Train the model
def train_model(model, optimizer, scheduler, earlystopping, num_epochs, t, tensor_data, index, loss_history, lambda_reg=1e-4):
    """Train the model."""
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        model_output = model(t)
        
        # Compute the loss function
        loss = einn_loss(model_output, tensor_data, model, t, train_size, model, lambda_reg)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record the loss
        loss_history.append(loss.item())
        
        # Early stopping
        earlystopping(loss.item(), model)
        if earlystopping.early_stop:
            print("Early stopping")
            break
        
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
            
    return loss_history, model

# Training with regularization
loss_history, model = train_model(model, optimizer, scheduler, earlystopping, num_epochs, t, tensor_data, index, loss_history, lambda_reg=1e-4)

# Function to plot the loss history
def plot_loss(losses, title="Training Loss", filename=None):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(losses) + 1), losses, label='Loss', color='black')
    plt.yscale('log')
    plt.title(f"{title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, format='pdf', dpi=600)
    plt.show()

# Function to plot the actual vs predicted values
def plot_actual_vs_predicted(data, actual_values, predicted_values, features, train_size, areaname, filename=None):
    fig, ax = plt.subplots(len(features), 1, figsize=(10, 12), sharex=True)

    for i, feature in enumerate(features):
        ax[i].plot(data["date"], actual_values[i], label="Actual", color="black", marker="o", markersize=3, linestyle='None')
        ax[i].plot(data["date"], predicted_values[i], label="Predicted", color="red", linestyle="--", linewidth=2)
        ax[i].axvline(data["date"].iloc[train_size], color="blue", linestyle="--", label="Train size")
        ax[i].set_ylabel(feature.replace("_", " ").title())
        ax[i].legend()
        ax[i].grid(True, linestyle='--', alpha=0.7)

    ax[-1].set_xlabel("Date")
    plt.suptitle(f"EINN Model Predictions for {areaname}")
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if filename:
        plt.savefig(filename, format='pdf', dpi=600)
    plt.show()


# Generate predictions
model.eval()
with torch.no_grad():
    model_output = model(t).cpu().numpy()
    
# Split the model output into the different compartments
S_pred, E_pred, Ia_pred, Is_pred, H_pred, C_pred, R_pred, D_pred = np.split(model_output, 8, axis=1)

# Extract training and validation data from tensor_data
Is_train, H_train, C_train, D_train, R_train = tensor_data["train"]
Is_val, H_val, C_val, D_val, R_val = tensor_data["val"]

# Normalize the data
N = 1

# Scale the predictions back to the original scale
scaler.fit(data[features])
scaled_predictions = np.concatenate([Is_pred, H_pred, C_pred, D_pred, R_pred], axis=1)
original_scale_predictions = scaler.inverse_transform(scaled_predictions)

Is_pred, H_pred, C_pred, D_pred, R_pred = np.split(original_scale_predictions, 5, axis=1)

# actual values
actual_values = data[features].values
Is_actual, H_actual, C_actual, D_actual, R_actual = np.split(actual_values, 5, axis=1)

# Convert actual and predicted values to lists of arrays for easier plotting
actual_values = [Is_actual, H_actual, C_actual, D_actual, R_actual]
predicted_values = [Is_pred, H_pred, C_pred, D_pred, R_pred]


# Plot loss history
plot_loss(loss_history, title="EINN Training Loss", filename=f"../../reports/figures/loss/{train_size}_{areaname}_loss_history.pdf")

# Plot actual vs predicted values
plot_actual_vs_predicted(data, actual_values, predicted_values, features, train_size, areaname, filename=f"../../reports/figures/Pinn/{train_size}_{areaname}_predictions.pdf")


# Calculate and print the metrics for each state
metrics = {}
metrics["Is"] = calculate_all_metrics(Is_actual, Is_pred, Is_train.cpu().numpy(), "New Confirmed", train_size, areaname)
metrics["H"] = calculate_all_metrics(H_actual, H_pred, H_train.cpu().numpy(), "New Admissions", train_size, areaname)
metrics["C"] = calculate_all_metrics(C_actual, C_pred, C_train.cpu().numpy(), "Occupied MV Beds", train_size, areaname)
metrics["D"] = calculate_all_metrics(D_actual, D_pred, D_train.cpu().numpy(), "New Deceased", train_size, areaname)
metrics["R"] = calculate_all_metrics(R_actual, R_pred, R_train.cpu().numpy(), "Recovered", train_size, areaname)

# save the metrics as csv
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(f"../../reports/results/{train_size}_{areaname}_metrics.csv", index=False)
# extract the learned parameters
beta = model.beta.cpu().item()
omega = model.omega.cpu().item()
mu = model.delta.cpu().item()
gamma_c = model.gamma_c.cpu().item()
delta_c = model.delta_c.cpu().item()

# Print the learned parameters
print(f"Learned parameters:")
print(f"Beta: {beta:.4f}")
print(f"Omega: {omega:.4f}")
print(f"Mu: {mu:.4f}")
print(f"Gamma_c: {gamma_c:.4f}")
print(f"Delta_c: {delta_c:.4f}")

# save as csv
learned_params = pd.DataFrame({
    "beta": [beta],
    "omega": [omega],
    "mu": [mu],
    "gamma_c": [gamma_c],
    "delta_c": [delta_c]
})

learned_params.to_csv(f"../../reports/parameters/{train_size}_{areaname}_learned_params.csv", index=False)
