import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm
from collections import deque

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
    df = df[df["nhs_region"] == areaname].reset_index(drop=True)
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
data = load_and_preprocess_data("../../data/hos_data/merged_data.csv", areaname="South West", recovery_period=21, rolling_window=7, start_date="2020-04-01", end_date="2020-08-31")

data.head(10)

def SIHCRD_model(t, y, beta, gamma, delta, rho, eta, kappa, mu, xi, N):
    S, I, H, C, R, D = y
    dSdt = -(beta * I / N) * S
    dIdt = (beta * S / N) * I - (gamma + rho + delta) * I
    dHdt = rho * I - (eta + kappa) * H
    dCdt = eta * H - (mu + xi) * C
    dRdt = gamma * I + kappa * H + mu * C
    dDdt = delta * I + xi * C
    return [dSdt, dIdt, dHdt, dCdt, dRdt, dDdt]

def prepare_tensors(data, device):
    t = tensor(range(1, len(data) + 1), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
    S = tensor(data["S(t)"].values, dtype=torch.float32).view(-1, 1).to(device)
    I = tensor(data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
    R = tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = tensor(data["new_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
    H = tensor(data["hospitalCases"].values, dtype=torch.float32).view(-1, 1).to(device)
    C = tensor(data["covidOccupiedMVBeds"].values, dtype=torch.float32).view(-1, 1).to(device)
    return t, S, I, R, D, H, C

def split_and_scale_data(data, train_size, features, device):
    scaler = MinMaxScaler()
    scaler.fit(data[features])

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:]

    scaled_train_data = pd.DataFrame(scaler.transform(train_data[features]), columns=features)
    scaled_val_data = pd.DataFrame(scaler.transform(val_data[features]), columns=features)

    # Prepare tensors for training and validation
    t_train, S_train, I_train, R_train, D_train, H_train, C_train = prepare_tensors(scaled_train_data, device)
    t_val, S_val, I_val, R_val, D_val, H_val, C_val = prepare_tensors(scaled_val_data, device)

    tensor_data = {
        "train": (t_train, S_train, I_train, R_train, D_train, H_train, C_train),
        "val": (t_val, S_val, I_val, R_val, D_val, H_val, C_val),
    }

    return tensor_data, scaler

features = ["S(t)", "active_cases", "hospitalCases", "covidOccupiedMVBeds", "recovered", "new_deceased"]
train_size = 60  # days

class StateNN(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=6):
        super(StateNN, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        # Initialize layers array starting with input layer
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        # Append hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])

        # Append output layer
        layers.append(nn.Linear(hidden_neurons, output_size))  # Epidemiological outputs

        # Convert list of layers to nn.Sequential
        self.net = nn.Sequential(*layers)

        # Initialize weights
        self.init_xavier()

    def forward(self, t):
        return self.net(t)

    def init_xavier(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)
        self.net.apply(init_weights)

class ParamNN(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=8):
        super(ParamNN, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        # Initialize layers array starting with the input layer
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        # Append hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])

        # Append output layer
        layers.append(nn.Linear(hidden_neurons, output_size))  # Output layer for estimating parameters

        # Convert list of layers to nn.Sequential
        self.net = nn.Sequential(*layers)

        # Initialize weights
        self.init_xavier()

    def init_xavier(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)
        self.net.apply(init_weights)

    def forward(self, t):
        return self.net(t)

    def get_params(self, t):
        """ Get the time-varying parameters """
        param = self.forward(t)
        
        beta = torch.sigmoid(param[:, 0])
        gamma = torch.sigmoid(param[:, 1])
        delta = torch.sigmoid(param[:, 2])
        rho = torch.sigmoid(param[:, 3])
        eta = torch.sigmoid(param[:, 4])
        kappa = torch.sigmoid(param[:, 5])
        mu = torch.sigmoid(param[:, 6])
        xi = torch.sigmoid(param[:, 7])

        return [beta, gamma, delta, rho, eta, kappa, mu, xi]


# def network_prediction(t, model, device, scaler, N):
#     """Generate predictions from the SEIRDNet model."""
#     t_tensor = torch.from_numpy(t).float().view(-1, 1).to(device).requires_grad_(True)
#     with torch.no_grad():
#         predictions = model(t_tensor)
#         predictions = predictions.cpu().numpy()
#         predictions = scaler.inverse_transform(predictions)
#     return predictions

# Split and scale the data
tensor_data, scaler = split_and_scale_data(data, train_size, features, device)

def pinn_loss(t, data, state_nn, param_nn, N, train_size=None):
    """Physics-Informed Neural Network loss function."""
    states_pred = state_nn(t)
    S_pred, I_pred, R_pred, D_pred, H_pred, C_pred = states_pred[:, 0], states_pred[:, 1], states_pred[:, 2], states_pred[:, 3], states_pred[:, 4], states_pred[:, 5]

    S_train, I_train, R_train, D_train, H_train, C_train = data["train"][1:]
    S_val, I_val, R_val, D_val, H_val, C_val = data["val"][1:]

    S_total = torch.cat([S_train, S_val])
    I_total = torch.cat([I_train, I_val])
    R_total = torch.cat([R_train, R_val])
    D_total = torch.cat([D_train, D_val])
    H_total = torch.cat([H_train, H_val])
    C_total = torch.cat([C_train, C_val])

    S_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    I_t = grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    R_t = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    D_t = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]
    H_t = grad(H_pred, t, grad_outputs=torch.ones_like(H_pred), create_graph=True)[0]
    C_t = grad(C_pred, t, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]

    beta_pred, gamma_pred, delta_pred, rho_pred, eta_pred, kappa_pred, mu_pred, xi_pred = param_nn.get_params(t)

    dSdt, dIdt, dHdt, dCdt, dRdt, dDdt = SIHCRD_model(t, [S_pred, I_pred, H_pred, C_pred, R_pred, D_pred], beta_pred, gamma_pred, delta_pred, rho_pred, eta_pred, kappa_pred, mu_pred, xi_pred, N)

    if train_size is not None:
        index = torch.randperm(train_size)
    else:
        index = torch.arange(len(t))

    data_loss = torch.mean((S_pred[index] - S_total[index]) ** 2) + torch.mean((I_pred[index] - I_total[index]) ** 2) + torch.mean((R_pred[index] - R_total[index]) ** 2) + torch.mean((D_pred[index] - D_total[index]) ** 2) + torch.mean((H_pred[index] - H_total[index]) ** 2) + torch.mean((C_pred[index] - C_total[index]) ** 2)
    derivatives_loss = torch.mean((S_t - dSdt) ** 2) + torch.mean((I_t - dIdt) ** 2) + torch.mean((R_t - dRdt) ** 2) + torch.mean((D_t - dDdt) ** 2) + torch.mean((H_t - dHdt) ** 2) + torch.mean((C_t - dCdt) ** 2)

    S0, I0, R0, D0, H0, C0 = S_pred[0], I_pred[0], R_pred[0], D_pred[0], H_pred[0], C_pred[0]
    initial_condition_loss = torch.mean((S_pred[0] - S0) ** 2) + torch.mean((I_pred[0] - I0) ** 2) + torch.mean((R_pred[0] - R0) ** 2) + torch.mean((D_pred[0] - D0) ** 2) + torch.mean((H_pred[0] - H0) ** 2) + torch.mean((C_pred[0] - C0) ** 2)

    total_loss = data_loss + derivatives_loss + initial_condition_loss

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
N = data["population"].values[0]

# Instantiate the neural networks
state_nn = StateNN(num_layers=6, hidden_neurons=32).to(device)
param_nn = ParamNN(num_layers=6, hidden_neurons=32).to(device)

# Optimizers
optimizer_state = optim.Adam(state_nn.parameters(), lr=learning_rate)
optimizer_param = optim.Adam(param_nn.parameters(), lr=learning_rate)

# Early stopping criteria
early_stopping = EarlyStopping(patience=10, verbose=False)

# Set the number of epochs for training
epochs = 50000

# Full time input for the entire dataset
t = (
    torch.tensor(np.arange(len(data)), dtype=torch.float32)
    .view(-1, 1)
    .to(device)
    .requires_grad_(True)
)

# Shuffle the data index
index = torch.randperm(len(tensor_data["train"][0]))

# Training loop function
def train_model(epochs, t, data, state_nn, param_nn, optimizer_state, optimizer_param, N, early_stopping, index):
    loss_history = []
    for epoch in tqdm(range(epochs)):
        state_nn.train()
        param_nn.train()

        optimizer_state.zero_grad()
        optimizer_param.zero_grad()

        index = torch.randperm(len(data["train"][0]))
        loss = pinn_loss(t, data, state_nn, param_nn, N, train_size=len(index))
        loss.backward()

        optimizer_state.step()
        optimizer_param.step()

        loss_history.append(loss.item())

        if early_stopping(loss.item()):
            print(f"Early stopping at epoch {epoch}. No improvement in loss for {early_stopping.patience} epochs.")
            break

        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    return loss_history, state_nn, param_nn

# Train the model
loss_history, state_nn, param_nn = train_model(epochs, t, tensor_data, state_nn, param_nn, optimizer_state, optimizer_param, N, early_stopping, index)

# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(np.log10(loss_history), label="Training Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Function to generate predictions for the entire dataset
def generate_predictions(t, state_nn, param_nn, device, scaler):
    state_nn.eval()
    param_nn.eval()

    with torch.no_grad():
        # Generate state predictions
        states_pred = state_nn(t)
        S_pred, I_pred, R_pred, D_pred, H_pred, C_pred = states_pred[:, 0], states_pred[:, 1], states_pred[:, 2], states_pred[:, 3], states_pred[:, 4], states_pred[:, 5]

        # Generate parameter predictions
        params_pred = param_nn(t)
        beta_pred, gamma_pred, delta_pred, rho_pred, eta_pred, kappa_pred, mu_pred, xi_pred = params_pred.transpose(0, 1)

        # Convert predictions to numpy arrays
        states_pred = states_pred.cpu().numpy()
        params_pred = torch.stack([beta_pred, gamma_pred, delta_pred, rho_pred, eta_pred, kappa_pred, mu_pred, xi_pred], dim=1).cpu().numpy()

        # Inverse transform the state predictions to original scale
        states_pred = scaler.inverse_transform(states_pred)

    return states_pred, params_pred

# Generate predictions
states_pred, params_pred = generate_predictions(t, state_nn, param_nn, device, scaler)

# Extract individual state predictions
S_pred = states_pred[:, 0]
I_pred = states_pred[:, 1]
R_pred = states_pred[:, 2]
D_pred = states_pred[:, 3]
H_pred = states_pred[:, 4]
C_pred = states_pred[:, 5]

# Extract individual parameter predictions
beta_pred = params_pred[:, 0]
gamma_pred = params_pred[:, 1]
delta_pred = params_pred[:, 2]
rho_pred = params_pred[:, 3]
eta_pred = params_pred[:, 4]
kappa_pred = params_pred[:, 5]
mu_pred = params_pred[:, 6]
xi_pred = params_pred[:, 7]

# Detach the tensor from the computation graph and convert to numpy array
t_numpy = t.detach().cpu().numpy()
dates = data["date"]


# Actual data
S_actual = data["S(t)"].values
I_actual = data["active_cases"].values
R_actual = data["recovered"].values
D_actual = data["new_deceased"].values
H_actual = data["hospitalCases"].values
C_actual = data["covidOccupiedMVBeds"].values

# Define training index size
train_index_size = len(tensor_data["train"][0])

fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
# Define plot details
plot_details = [
    ("Susceptible", S_actual, S_pred),
    # ("Exposed", E_actual, E_pred),
    ("Active Cases", I_actual, I_pred),
    ("Recovered", R_actual, R_pred),
    ("Deceased", D_actual, D_pred),
    ("Hospitalized", H_actual, H_pred),
    ("ICU Beds", C_actual, C_pred),
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
plt.subplots_adjust(top=0.7)
plt.savefig(f"../../reports/figures/{train_size}_pinn_{areaname}_results.pdf")
plt.show()