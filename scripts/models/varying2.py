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
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from tqdm.autonotebook import tqdm

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


def load_and_preprocess_data(
    filepath,
    areaname,
    recovery_period=16,
    rolling_window=7,
    start_date="2020-04-01",
    end_date="2020-07-31",
):
    """Load and preprocess the data from a CSV file."""
    df = pd.read_csv(filepath)
    df = df[df["nhs_region"] == areaname].reset_index(drop=True)
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
    # susceptible population
    df["susceptible"] = df["population"] - df["recovered"] - df["cumulative_deceased"] - df["cumulative_confirmed"]
    
    cols_to_smooth = [
        "cumulative_confirmed",
        "cumulative_deceased",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "recovered",
        "active_cases",
        "new_deceased",
        "new_confirmed",
        "susceptible"
    ]
    
    # make the data are non-negative
    for col in cols_to_smooth:
        df[col] = df[col].clip(lower=0)
    
    # apply rolling window smoothing
    for col in cols_to_smooth:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    return df


# Load and preprocess the data
data = load_and_preprocess_data(
    "../../data/hos_data/merged_data.csv",
    areaname="South West",
    recovery_period=21,
    start_date="2021-01-01",
    end_date="2022-05-31",
)

days = len(data)

# plot the suscetible
plt.figure(figsize=(10, 5))
plt.plot(data["date"], data["susceptible"], label="Susceptible", color="blue")
plt.xlabel("Date")
plt.ylabel("Susceptible")
plt.title("Susceptible Population Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.show()


class SEIRDNet(nn.Module):
    """Epidemiological network for predicting SEIRD model outputs."""

    def __init__(self, num_layers=4, hidden_neurons=20):
        super(SEIRDNet, self).__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(
            nn.Linear(hidden_neurons, 5)  # Adjust the output size to 6 (S, I, R, D, H)
        )
        self.net = nn.Sequential(*layers)
        self.init_xavier()

    def forward(self, t):
        return self.net(t)

    def init_xavier(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.001)

        self.apply(init_weights)


class ParameterNet(nn.Module):
    """Network for estimating SEIRD model parameters."""

    def __init__(self, num_layers=4, hidden_neurons=20):
        super(ParameterNet, self).__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.ReLU()])
        layers.append(
            nn.Linear(hidden_neurons, 3)  # Adjust the output size to 3 (beta, gamma, delta)
        )
        self.net = nn.Sequential(*layers)
        self.init_xavier()

    def forward(self, t):
        params = self.net(t)
        beta = torch.sigmoid(params[:, 0])
        gamma = torch.sigmoid(params[:, 1])
        delta = torch.sigmoid(params[:, 2])
        return beta, gamma, delta

    def init_xavier(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("relu")
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.001)

        self.apply(init_weights)


def prepare_tensors(data, device):
    """Prepare tensors for training."""
    t = (
        tensor(range(1, len(data) + 1), dtype=torch.float32)
        .view(-1, 1)
        .to(device)
        .requires_grad_(True)
    )
    S = tensor(data["susceptible"].values, dtype=torch.float32).view(-1, 1).to(device)
    I = tensor(data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
    R = tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = tensor(data["new_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
    H = tensor(data["covidOccupiedMVBeds"].values, dtype=torch.float32).view(-1, 1).to(device)
    return t, S, I, R, D, H


def split_and_scale_data(data, train_size, features, device):
    """Split and scale data into training and validation sets."""
    scaler = MinMaxScaler()
    scaler.fit(data[features])

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:]

    scaled_train_data = pd.DataFrame(
        scaler.transform(train_data[features]), columns=features
    )
    scaled_val_data = pd.DataFrame(
        scaler.transform(val_data[features]), columns=features
    )

    t_train, S_train, I_train, R_train, D_train, H_train = prepare_tensors(scaled_train_data, device)
    t_val, S_val, I_val, R_val, D_val, H_val = prepare_tensors(scaled_val_data, device)

    tensor_data = {
        "train": (t_train, S_train, I_train, R_train, D_train, H_train),
        "val": (t_val, S_val, I_val, R_val, D_val, H_val),
    }

    return tensor_data, scaler


# Define features and data split
features = ["susceptible", "active_cases", "recovered", "new_deceased", "covidOccupiedMVBeds"]
train_size = 60  # days

# Split and scale data
tensor_data, scaler = split_and_scale_data(data, train_size, features, device)


# PINN loss function
def pinn_loss(tensor_data, params, model_output, t, index=None):
    """Physics-informed neural network loss function."""
    t_train, S_train, I_train, R_train, D_train, H_train = tensor_data["train"]
    t_val, S_val, I_val, R_val, D_val, H_val = tensor_data["val"]

    N = 1.0
    
    # total data
    S_total = torch.cat([S_train, S_val], dim=0)
    I_total = torch.cat([I_train, I_val], dim=0)
    R_total = torch.cat([R_train, R_val], dim=0)
    D_total = torch.cat([D_train, D_val], dim=0)
    H_total = torch.cat([H_train, H_val], dim=0)

    beta, gamma, delta = params
    S_pred, I_pred, R_pred, D_pred, H_pred = (
        model_output[:, 0],
        model_output[:, 1],
        model_output[:, 2],
        model_output[:, 3],
        model_output[:, 4],
    )

    # Compute the gradients
    S_grad = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    # E_grad = grad(E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True)[0]
    I_grad = grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    R_grad = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    D_grad = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]
    H_grad = grad(H_pred, t, grad_outputs=torch.ones_like(H_pred), create_graph=True)[0]

    # solve the SEIRD model
    dsDt = -beta * S_pred * I_pred / N
    dIdt = beta * S_pred * I_pred / N - gamma * I_pred
    dRdt = gamma * I_pred
    dDdt = delta * I_pred
    dHdt = gamma * I_pred
    

    # Generate a random subset of indices if provided
    if index is not None:
        S_pred, I_pred, R_pred, D_pred, H_pred = S_pred[index], I_pred[index], R_pred[index], D_pred[index], H_pred[index]
        S_total, I_total, R_total, D_total, H_total = S_total[index], I_total[index], R_total[index], D_total[index], H_total[index]


    # data loss
    data_loss = (
        torch.mean((S_pred - S_total) ** 2)
        + torch.mean((I_pred - I_total) ** 2)
        + torch.mean((R_pred - R_total) ** 2)
        + torch.mean((D_pred - D_total) ** 2)
        + torch.mean((H_pred - H_total) ** 2)
    )

    # Differential equation residuals
    residual_loss = (
        torch.mean((S_grad + dsDt) ** 2)
        + torch.mean((I_grad + dIdt) ** 2)
        + torch.mean((R_grad + dRdt) ** 2)
        + torch.mean((D_grad + dDdt) ** 2)
        + torch.mean((H_grad + dHdt) ** 2)
    )

    # initial condition loss
    S0, I0, R0, D0, H0 = S_pred[0], I_pred[0], R_pred[0], D_pred[0], H_pred[0]
    initial_loss = (
        torch.mean((S_pred[0] - S0) ** 2)
        + torch.mean((I_pred[0] - I0) ** 2)
        + torch.mean((R_pred[0] - R0) ** 2)
        + torch.mean((D_pred[0] - D0) ** 2)
        + torch.mean((H_pred[0] - H0) ** 2)
    )
    
    # regularise the loss with alpha
    alpha = 1e-4
    loss = data_loss + alpha * residual_loss + alpha * initial_loss
    
    

    return loss


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


# Initialize the models
model = SEIRDNet(num_layers=10, hidden_neurons=32).to(device)
param_net = ParameterNet(num_layers=10, hidden_neurons=32).to(device)

# Initialize the optimizers
model_optimizer = optim.Adam(model.parameters(), lr=1e-3)
param_optimizer = optim.Adam(param_net.parameters(), lr=1e-3)

# Learning rate scheduler
scheduler = StepLR(model_optimizer, step_size=5000, gamma=0.7)

# Early stopping criteria
early_stopping = EarlyStopping(patience=100, verbose=False)

# Loss history
loss_history = []

# Total population
N = data["population"].values[0]  # Assuming the population is constant and given in the data


def train_model(
    tensor_data,
    model,
    param_net,
    model_optimizer,
    param_optimizer,
    scheduler,
    early_stopping,
    num_epochs=1000,
):
    """Training loop for the model."""
    for epoch in tqdm(range(num_epochs)):
        model.train()
        param_net.train()

        model_optimizer.zero_grad()
        param_optimizer.zero_grad()

        t = (
            torch.tensor(np.arange(len(data)), dtype=torch.float32)
            .view(-1, 1)
            .to(device)
            .requires_grad_(True)
        )
        
        # Shuffle the training index for each epoch
        index = torch.randperm(len(tensor_data["train"][0]))
        
        # Predict the SEIRD model outputs
        params = param_net(t)
        model_output = model(t)
        
        # Compute the loss
        loss = pinn_loss(tensor_data, params, model_output, t, index)
        
        loss.backward()

        model_optimizer.step()
        param_optimizer.step()

        scheduler.step()

        loss_history.append(loss.item())
        
        # Early stopping
        early_stopping(loss.item())
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}: Loss: {loss.item():.6f}")

    print("Finished Training")
    
    return loss_history, model, param_net


# Train the model
loss_history, model, param_net = train_model(
    tensor_data,
    model,
    param_net,
    model_optimizer,
    param_optimizer,
    scheduler,
    early_stopping,
    num_epochs=50000,
)

# Plot training loss in log scale
plt.figure(figsize=(10, 5))
plt.plot(np.log10(loss_history), label="Training Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.title("Training Loss over Epochs (Log Scale)")
plt.legend()
plt.show()



# Total time frame from the start of the training data to the end of the validation data
t_total = np.arange(len(data))


# Predict the SEIRD model outputs
params = param_net(torch.from_numpy(t_total).float().view(-1, 1).to(device))
model_output = model(torch.from_numpy(t_total).float().view(-1, 1).to(device))

# plot model outputs for H_pred
plt.figure(figsize=(10, 5))
plt.plot(model_output.detach().cpu().numpy()[:, 4], label="ICU Bed Demand (Model)", color="red"
)
plt.xlabel("Days")
plt.ylabel("ICU Bed Demand")
plt.title("ICU Bed Demand Prediction")
plt.legend()
plt.show()

# Inverse transform the scaled data
predicted_data = scaler.inverse_transform(model_output.detach().cpu().numpy())




# Create a DataFrame for the predicted data
S_pred, I_pred, R_pred, D_pred, H_pred = (
    predicted_data[:, 0],
    predicted_data[:, 1],
    predicted_data[:, 2],
    predicted_data[:, 3],
    predicted_data[:, 4],
)


# plot the results
plt.figure(figsize=(10, 5))
plt.plot(data["date"], data["covidOccupiedMVBeds"], label="ICU Bed Demand (Data)", color="blue")
plt.plot(data["date"], H_pred, label="ICU Bed Demand (Model)", color="red")
plt.scatter(data["date"][:train_size], data["covidOccupiedMVBeds"][:train_size], color="black", label="Training Data")
plt.xlabel("Date")
plt.ylabel("ICU Bed Demand")
plt.title("ICU Bed Demand Prediction")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Calculate evaluation metrics
def evaluate_metrics(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    nrmse = np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))
    return mape, nrmse

# Calculate MAPE and NRMSE for the ICU bed demand prediction
mape, nrmse = evaluate_metrics(data["covidOccupiedMVBeds"].values, H_pred)
print(f"ICU Bed Demand - MAPE: {mape:.4f}, NRMSE: {nrmse:.4f}")


# Conduct an additional experiment for comparison (e.g., a simpler model)
# Simple Linear Regression model for comparison
from sklearn.linear_model import LinearRegression

# Prepare the data for linear regression
t_values = np.arange(len(data)).reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(t_values[:train_size], data["covidOccupiedMVBeds"][:train_size])
linear_predictions = linear_regressor.predict(t_values)

# Calculate MAPE and NRMSE for the linear regression model
mape_lr, nrmse_lr = evaluate_metrics(data["covidOccupiedMVBeds"].values, linear_predictions)
print(f"Linear Regression - MAPE: {mape_lr:.4f}, NRMSE: {nrmse_lr:.4f}")

# Plot comparison results
plt.figure(figsize=(10, 5))
plt.plot(data["date"], data["covidOccupiedMVBeds"], label="ICU Bed Demand (Data)", color="blue")
plt.plot(data["date"], predicted_icu_beds, label="ICU Bed Demand (PINN Model)", color="red")
plt.plot(data["date"], linear_predictions, label="ICU Bed Demand (Linear Regression)", color="green")
plt.xlabel("Date")
plt.ylabel("ICU Bed Demand")
plt.title("ICU Bed Demand Prediction Comparison")
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Calculate effective reproduction number (Rc)
def calculate_reproduction_number(beta, rho, ds, kappa, da, alpha):
    return beta * (rho * ds + kappa * (1 - rho) * da + kappa / alpha)

# Example values for rho, ds, kappa, da, and alpha
rho = 0.2
ds = 10
kappa = 0.5
da = 7
alpha = 0.1

# Calculate Rc for each time step
beta = param_net(torch.from_numpy(t_total).float().view(-1, 1).to(device))[0].detach().cpu().numpy()
Rc = calculate_reproduction_number(beta, rho, ds, kappa, da, alpha)

# Plot the effective reproduction number over time
plt.figure(figsize=(10, 5))
plt.plot(data["date"], Rc, label="Effective Reproduction Number (Rc)", color="purple")
plt.xlabel("Date")
plt.ylabel("Rc")
plt.title("Effective Reproduction Number (Rc) Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.show()
