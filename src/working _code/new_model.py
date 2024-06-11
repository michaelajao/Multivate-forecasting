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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

# Ensure necessary directories exist
os.makedirs("../../models", exist_ok=True)
os.makedirs("../../reports/figures", exist_ok=True)
os.makedirs("../../reports/results", exist_ok=True)
os.makedirs("../../reports/England", exist_ok=True)

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

# Set matplotlib style and parameters
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {   "font.family": "serif",
        "font.size": 14,
        "figure.figsize": [8, 5],
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
        "axes.grid": False,
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
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (
        np.max(y_true) - np.min(y_true)
    )


def safe_mean_absolute_scaled_error(y_true, y_pred, y_train, epsilon=1e-10):
    """Calculate the Mean Absolute Scaled Error (MASE) safely."""
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    d = max(d, epsilon)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d


def calculate_errors(y_true, y_pred, y_train, areaname):
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


def calculate_all_metrics(actual, predicted, train_data, label, areaname):
    """Calculate metrics for each state."""
    print(f"\nMetrics for {label}:")
    mape, nrmse, mase, rmse, mae, mse = calculate_errors(
        actual, predicted, train_data, areaname
    )
    return mape, nrmse, mase, rmse, mae, mse


# Define the SEIRD model differential equations
def seird_model(
    y, t, N, beta, alpha, rho, ds, da, omega, dH, mu, gamma_c, delta_c, eta
):
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

# def load_preprocess_data(
#     filepath, areaname, recovery_period=16, rolling_window=7, end_date=None
# ):
#     """Load and preprocess the COVID-19 data."""
#     df = pd.read_csv(filepath)

#     # Convert the date column to datetime
#     df["date"] = pd.to_datetime(df["date"])

#     # Select data up to the end_date
#     if end_date:
#         df = df[df["date"] <= end_date]

#     cols_to_smooth = [
#         "cumulative_confirmed",
#         "cumulative_deceased",
#         "hospitalCases",
#         "covidOccupiedMVBeds",
#         "new_deceased",
#         "new_confirmed",
#         "newAdmissions",
#     ]
#     for col in cols_to_smooth:
#         df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

#     return df

def load_and_preprocess_data(filepath, areaname, rolling_window=7, start_date="2020-04-01", end_date="2021-08-31"):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    
    # Compute daily new values from cumulative values
    df['daily_confirmed'] = df["cumulative_confirmed"].diff().fillna(0)
    df['daily_deceased'] = df["cumulative_deceased"].diff().fillna(0)
    df['daily_hospitalized'] = df["cumAdmissions"].diff().fillna(0)
    
    # Ensure no negative values
    df['daily_confirmed'] = df['daily_confirmed'].clip(lower=0)
    df['daily_deceased'] = df['daily_deceased'].clip(lower=0)
    df['daily_hospitalized'] = df['daily_hospitalized'].clip(lower=0)
    
    required_columns = ["date","population", "cumulative_confirmed", "cumulative_deceased", "new_confirmed", "new_deceased", "cumAdmissions", "daily_confirmed", "daily_deceased", "daily_hospitalized", "hospitalCases", "covidOccupiedMVBeds", "newAdmissions"]
    
    # Select required columns
    df = df[required_columns]
    
    # Apply 7-day rolling average to smooth out data (except for date and population)
    for col in required_columns[2:]:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)
        
    # Select data from start date to end date
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df.loc[mask]
    
    return df

data = load_and_preprocess_data(
    "../../data/processed/england_data.csv",
    "England",
    rolling_window=7,
    start_date="2020-04-01", end_date="2020-12-31"
)


# Print the columns to ensure the required columns are created
print("DataFrame columns:", data.columns)

plt.plot(data["date"], data["daily_deceased"])
plt.title("New Deceased over time")
plt.xlabel("Date")
plt.ylabel("New Deceased")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


def prepare_tensors(data, device):
    """Prepare tensors for training."""
    I = torch.tensor(data["daily_confirmed"].values, dtype=torch.float32).view(-1, 1).to(device)
    H = torch.tensor(data["daily_hospitalized"].values, dtype=torch.float32).view(-1, 1).to(device)
    C = torch.tensor(data["covidOccupiedMVBeds"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = torch.tensor(data["daily_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
    return I, H, C, D


def scale_data(data, features, device):
    """Split and scale data into training and validation sets."""
    scaler = MinMaxScaler()
    scaler.fit(data[features])

    # Scale the data
    data_scaled = scaler.transform(data[features])

    # Convert the data to PyTorch tensors
    data_scaled = torch.tensor(data_scaled, dtype=torch.float32).to(device)

    return data_scaled, scaler


features = [
    "daily_confirmed",
    "daily_hospitalized",
    "covidOccupiedMVBeds",
    "daily_deceased",
]

# Split and scale the data
data_scaled, scaler = scale_data(data, features, device)


class EpiNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=8):
        super(EpiNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        # Input layer
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])

        # Output layer
        layers.append(nn.Linear(hidden_neurons, output_size))
        self.net = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        self.init_xavier()

    def forward(self, t):
        return torch.sigmoid(self.net(t))

    def init_xavier(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        # Apply the weight initialization to the network
        self.net.apply(init_weights)


class ParameterNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10):
        super(ParameterNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])

        layers.append(nn.Linear(hidden_neurons, 6))
        self.net = nn.Sequential(*layers)

        self.init_xavier()

    def forward(self, t):
        return self.net(t)
    
    def get_parameters(self, t):
        raw_parameters = self.net(t)
        
        # Apply the sigmoid function to ensure the parameters are in the correct range
        beta = torch.sigmoid(raw_parameters[:, 0]) * 0.5
        omega = torch.sigmoid(raw_parameters[:, 1]) * 0.1
        mu = torch.sigmoid(raw_parameters[:, 2]) * 0.1
        gamma_c = torch.sigmoid(raw_parameters[:, 3]) * 0.1
        delta_c = torch.sigmoid(raw_parameters[:, 4]) * 0.1
        eta = torch.sigmoid(raw_parameters[:, 5]) * 0.1

        return beta, omega, mu, gamma_c, delta_c, eta
    
    def init_xavier(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        # Apply the weight initialization to the network
        self.net.apply(init_weights)
        

def einn_loss(model_output, tensor_data, parameters, t):
    """Compute the loss function for the EINN model with L2 regularization."""

    # Split the model output into the different compartments
    S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred = (
        model_output[:, 0],
        model_output[:, 1],
        model_output[:, 2],
        model_output[:, 3],
        model_output[:, 4],
        model_output[:, 5],
        model_output[:, 6],
        model_output[:, 7],
    )
    

    # Normalize the data
    N = 1

    Is_data, H_data, C_data, D_data = (
        tensor_data[:, 0],
        tensor_data[:, 1],
        tensor_data[:, 2],
        tensor_data[:, 3],
    )

    # Constants based on the table provided
    rho = 0.80  # Proportion of symptomatic infections
    alpha = 1 / 5  # Incubation period (5 days)
    ds = 1 / 4  # Infectious period for symptomatic (4 days)
    da = 1 / 7  # Infectious period for asymptomatic (7 days)
    dH = 1 / 13.4  # Hospitalization days (13.4 days)

    # Learned parameters
    beta_pred, omega_pred, mu_pred, gamma_c_pred, delta_c_pred, eta_pred = parameters

    # Compute the differential equations
    S_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True, retain_graph=True)[0]
    E_t = grad(E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True, retain_graph=True)[0]
    Ia_t = grad(Ia_pred, t, grad_outputs=torch.ones_like(Ia_pred), create_graph=True, retain_graph=True)[0]
    Is_t = grad(Is_pred, t, grad_outputs=torch.ones_like(Is_pred), create_graph=True, retain_graph=True)[0]
    H_t = grad(H_pred, t, grad_outputs=torch.ones_like(H_pred), create_graph=True, retain_graph=True)[0]
    C_t = grad(C_pred, t, grad_outputs=torch.ones_like(C_pred), create_graph=True, retain_graph=True)[0]
    R_t = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True, retain_graph=True)[0]
    D_t = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True, retain_graph=True)[0]
    
    dSdt, dEdt, dIadt, dIsdt, dHdt, dCdt, dRdt, dDdt = seird_model(
        [S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred],
        t,
        N,
        beta_pred,
        alpha,
        rho,
        ds,
        da,
        omega_pred,
        dH,
        mu_pred,
        gamma_c_pred,
        delta_c_pred,
        eta_pred,
    )

    # Compute the loss function
    data_loss = (
        torch.mean((Is_pred - Is_data) ** 2)
        + torch.mean((H_pred - H_data) ** 2)
        + torch.mean((C_pred - C_data) ** 2)
        + torch.mean((D_pred - D_data) ** 2)
    )

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

    loss = data_loss + residual_loss
    return loss


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0

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


# Training loop
def train_model(model, parameter_net, optimizer, scheduler, time_stamps, data_scaled, num_epochs=100, early_stopping=None):
    train_losses = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        parameter_net.train()
        
        train_loss = 0.0

        t = time_stamps.to(device).float()
        data = data_scaled.to(device).float()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        model_output = model(t)
        parameters = parameter_net.get_parameters(t)

        # Compute loss
        loss = einn_loss(model_output, data, parameters, t)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        
        train_losses.append(train_loss)

        scheduler.step(train_loss)
        
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")

        # Check early stopping
        if early_stopping:
            early_stopping(train_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return train_losses


# Prepare data for training
data = load_preprocess_data(
    "../../data/processed/england_data.csv",
    "England",
    recovery_period=21,
    rolling_window=7,
    end_date="2021-12-31",
)

scaled_data, scaler = scale_data(data, features, device)

# Initialize model, optimizer, and scheduler
model = EpiNet(num_layers=5, hidden_neurons= 20, output_size=8).to(device)
parameter_net = ParameterNet(num_layers=2, hidden_neurons=20).to(device)
optimizer = optim.Adam(list(model.parameters()) + list(parameter_net.parameters()), lr=1e-4)
scheduler = StepLR(optimizer, step_size=5000, gamma=0.9)

# Early stopping
early_stopping = EarlyStopping(patience=100, verbose=True)

# Create timestamps tensor
time_stamps = torch.tensor(data.index.values, dtype=torch.float32).view(-1, 1).to(device).requires_grad_()

# Train the model
train_losses = train_model(model, parameter_net, optimizer, scheduler, time_stamps, scaled_data, num_epochs=50000, early_stopping=early_stopping)

# Plot training loss
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.yscale('log')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('../../reports/figures/training_loss.pdf')
plt.show()

# Save the trained model
torch.save(model.state_dict(), "../../models/epinet_model.pth")
torch.save(parameter_net.state_dict(), "../../models/parameter_net.pth")


# plot the outputs
def plot_outputs(model, parameter_net, data, device, scaler):
    model.eval()
    parameter_net.eval()

    with torch.no_grad():
        time_stamps = torch.tensor(data.index.values, dtype=torch.float32).view(-1, 1).to(device)
        model_output = model(time_stamps)
        parameters = parameter_net.get_parameters(time_stamps)

    # Extract only the observed outputs (columns 2, 4, 5, 7) for inverse scaling
    observed_model_output = model_output[:, [2, 4, 5, 7]].cpu().numpy()
    observed_model_output_scaled = scaler.inverse_transform(observed_model_output)

    dates = data["date"]

    # Plot observed vs. predicted outputs in a 1x4 grid
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), sharex=True)

    axs[0].plot(dates, data["daily_confirmed"], label="Observed Infections", color='blue')
    axs[0].plot(dates, observed_model_output_scaled[:, 0], label="Predicted Infections", linestyle='--', color='red')
    axs[0].set_ylabel("New Confirmed Cases")

    axs[1].plot(dates, data["daily_hospitalized"], label="Observed Hospitalizations", color='blue')
    axs[1].plot(dates, observed_model_output_scaled[:, 1], label="Predicted Hospitalizations", linestyle='--', color='red')
    axs[1].set_ylabel("New Admissions")

    axs[2].plot(dates, data["covidOccupiedMVBeds"], label="Observed Critical", color='blue')
    axs[2].plot(dates, observed_model_output_scaled[:, 2], label="Predicted Critical", linestyle='--', color='red')
    axs[2].set_ylabel("Critical Cases")


    axs[3].plot(dates, data["daily_deceased"], label="Observed Deaths", color='blue')
    axs[3].plot(dates, observed_model_output_scaled[:, 3], label="Predicted Deaths", linestyle='--', color='red')
    axs[3].set_ylabel("New Deaths")

    for ax in axs:
        ax.set_xlabel("Date")
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.legend()
    plt.savefig('../../reports/figures/observed_vs_predicted.pdf')
    plt.show()

    # Plot unobserved outputs in a 1x4 grid
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), sharex=True)

    axs[0].plot(dates, model_output[:, 0].cpu(), label="Susceptible", color='green')
    axs[0].set_ylabel("Susceptible")

    axs[1].plot(dates, model_output[:, 1].cpu(), label="Exposed", color='green')
    axs[1].set_ylabel("Exposed")

    axs[2].plot(dates, model_output[:, 3].cpu(), label="Asymptomatic Infected", color='green')
    axs[2].set_ylabel("Asymptomatic Infected")

    axs[3].plot(dates, model_output[:, 6].cpu(), label="Recovered", color='green')
    axs[3].set_ylabel("Recovered")

    for ax in axs:
        ax.set_xlabel("Date")
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig('../../reports/figures/unobserved_outputs.pdf')
    plt.show()

    # Plot time-varying parameters in a 2 * 3 grid
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    parameters_np = [p.cpu().numpy() for p in parameters]

    axs[0, 0].plot(dates, parameters_np[0], label="Beta (Infection rate)", color='purple')
    axs[0, 0].set_ylabel("Beta")
    axs[0, 0].legend()

    axs[0, 1].plot(dates, parameters_np[1], label="Omega (Symptomatic proportion)", color='purple')
    axs[0, 1].set_ylabel("Omega")
    axs[0, 1].legend()

    axs[0, 2].plot(dates, parameters_np[2], label="Mu (Mortality rate)", color='purple')
    axs[0, 2].set_ylabel("Mu")
    axs[0, 2].legend()

    axs[1, 0].plot(dates, parameters_np[3], label="Gamma_c (Critical recovery rate)", color='purple')
    axs[1, 0].set_ylabel("Gamma_c")
    axs[1, 0].legend()

    axs[1, 1].plot(dates, parameters_np[4], label="Delta_c (Critical mortality rate)", color='purple')
    axs[1, 1].set_ylabel("Delta_c")
    axs[1, 1].legend()

    axs[1, 2].plot(dates, parameters_np[5], label="Eta (Reinfection rate)", color='purple')
    axs[1, 2].set_ylabel("Eta")
    axs[1, 2].legend()


    for ax in axs.flat:
        ax.set_xlabel("Date")
        ax.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig('../../reports/figures/time_varying_parameters.pdf')
    plt.show()

# Plot the outputs and parameters
plot_outputs(model, parameter_net, data, device, scaler)

