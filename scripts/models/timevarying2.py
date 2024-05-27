import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import tensor, nn, optim
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
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
    # Print PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        # Print CUDA version and available GPUs
        print(f"CUDA version: {torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available. PyTorch will run on CPU.")

check_pytorch()

def load_and_preprocess_data(filepath, areaname, recovery_period=16, rolling_window=7, start_date="2020-04-01", end_date="2020-12-31"):
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
data = load_and_preprocess_data("../../data/hos_data/merged_data.csv", areaname="South West", recovery_period=21, start_date="2020-04-01")


class EpiNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=6, dropout_rate=0.2):
        super(EpiNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        # Initialize layers array starting with input layer
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh(), nn.Dropout(dropout_rate)]

        # Append hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh(), nn.Dropout(dropout_rate)])

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

class BetaNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=8, dropout_rate=0.2):
        super(BetaNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        # Initialize layers array starting with the input layer
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh(), nn.Dropout(dropout_rate)]

        # Append hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh(), nn.Dropout(dropout_rate)])

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
        params = self.net(t)
        # Beta (β) is a positive value between 0.1 and 1 using the sigmoid function
        beta = torch.sigmoid(params[:, 0]) * 0.9 + 0.1
        # Gamma (γ) is a positive value between 0.01 and 0.1 using the sigmoid function
        gamma = torch.sigmoid(params[:, 1]) * 0.1 + 0.01
        # Delta (δ) is a positive value between 0.001 and 0.01 using the sigmoid function
        delta = torch.sigmoid(params[:, 2]) * 0.01 + 0.001
        # Rho (ρ) is a positive value between 0.001 and 0.05 using the sigmoid function
        rho = torch.sigmoid(params[:, 3]) * 0.05 + 0.001
        # Eta (η) is a positive value between 0.001 and 0.05 using the sigmoid function
        eta = torch.sigmoid(params[:, 4]) * 0.05 + 0.001
        # Kappa (κ) is a positive value between 0.001 and 0.05 using the sigmoid function
        kappa = torch.sigmoid(params[:, 5]) * 0.05 + 0.001
        # Mu (μ) is a positive value between 0.001 and 0.05 using the sigmoid function
        mu = torch.sigmoid(params[:, 6]) * 0.05 + 0.001
        # Xi (ξ) is a positive value between 0.001 and 0.01 using the sigmoid function
        xi = torch.sigmoid(params[:, 7]) * 0.01 + 0.001
        return torch.stack([beta, gamma, delta, rho, eta, kappa, mu, xi], dim=1)


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

tensor_data, scaler = split_and_scale_data(data, train_size, features, device)

def pinn_loss(tensor_data, parameters, model_output, t, N, device):
    t_train, S_train, I_train, R_train, D_train, H_train, C_train = tensor_data["train"]
    t_val, S_val, I_val, R_val, D_val, H_val, C_val = tensor_data["val"]

    S = torch.cat([S_train, S_val], dim=0)
    I = torch.cat([I_train, I_val], dim=0)
    R = torch.cat([R_train, R_val], dim=0)
    D = torch.cat([D_train, D_val], dim=0)
    H = torch.cat([H_train, H_val], dim=0)
    C = torch.cat([C_train, C_val], dim=0)

    beta_pred = parameters[:, 0].squeeze()
    gamma_pred = parameters[:, 1].squeeze()
    delta_pred = parameters[:, 2].squeeze()
    rho_pred = parameters[:, 3].squeeze()
    eta_pred = parameters[:, 4].squeeze()
    kappa_pred = parameters[:, 5].squeeze()
    mu_pred = parameters[:, 6].squeeze()
    xi_pred = parameters[:, 7].squeeze()

    S_pred, I_pred, H_pred, C_pred, R_pred, D_pred = model_output.unbind(1)

    # Compute gradients
    s_t = torch.autograd.grad(outputs=S_pred, inputs=t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    i_t = torch.autograd.grad(outputs=I_pred, inputs=t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    h_t = torch.autograd.grad(outputs=H_pred, inputs=t, grad_outputs=torch.ones_like(H_pred), create_graph=True)[0]
    c_t = torch.autograd.grad(outputs=C_pred, inputs=t, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
    r_t = torch.autograd.grad(outputs=R_pred, inputs=t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    d_t = torch.autograd.grad(outputs=D_pred, inputs=t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]

    dSdt_pred, dIdt_pred, dHdt_pred, dCdt_pred, dRdt_pred, dDdt_pred = SIHCRD_model(
        t, [S_pred, I_pred, H_pred, C_pred, R_pred, D_pred],
        beta_pred, gamma_pred, delta_pred, rho_pred, eta_pred, kappa_pred, mu_pred, xi_pred, N
    )

    # Loss components
    data_loss = torch.mean((S - S_pred) ** 2 + (I - I_pred) ** 2 + (H - H_pred) ** 2 + (C - C_pred) ** 2 + (R - R_pred) ** 2 + (D - D_pred) ** 2)
    physics_loss = torch.mean((s_t - dSdt_pred) ** 2 + (i_t - dIdt_pred) ** 2 + (h_t - dHdt_pred) ** 2 + (c_t - dCdt_pred) ** 2 + (r_t - dRdt_pred) ** 2 + (d_t - dDdt_pred) ** 2)
    initial_condition_loss = torch.mean((S[0] - S_pred[0]) ** 2 + (I[0] - I_pred[0]) ** 2 + (H[0] - H_pred[0]) ** 2 + (C[0] - C_pred[0]) ** 2 + (R[0] - R_pred[0]) ** 2 + (D[0] - D_pred[0]) ** 2)
    boundary_condition_loss = torch.mean((S[-1] - S_pred[-1]) ** 2 + (I[-1] - I_pred[-1]) ** 2 + (H[-1] - H_pred[-1]) ** 2 + (C[-1] - C_pred[-1]) ** 2 + (R[-1] - R_pred[-1]) ** 2 + (D[-1] - D_pred[-1]) ** 2)
    reg_loss = torch.mean(beta_pred**2 + gamma_pred**2 + delta_pred**2 + rho_pred**2 + eta_pred**2 + kappa_pred**2 + mu_pred**2 + xi_pred**2)

    # Total loss
    loss = data_loss + physics_loss + initial_condition_loss + boundary_condition_loss + reg_loss

    return loss

class EarlyStopping:
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

# Initialize the models
model = EpiNet(num_layers=10, hidden_neurons=32, output_size=6, dropout_rate=0.2).to(device)
beta_net = BetaNet(num_layers=10, hidden_neurons=32, output_size=8, dropout_rate=0.2).to(device)

# Initialize the optimizers
model_optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Added L2 regularization
params_optimizer = optim.Adam(beta_net.parameters(), lr=1e-4, weight_decay=1e-5)  # Added L2 regularization

# Define the learning rate scheduler
model_scheduler = StepLR(model_optimizer, step_size=5000, gamma=0.998)
params_scheduler = StepLR(params_optimizer, step_size=5000, gamma=0.998)

# Early stopping criteria
early_stopping = EarlyStopping(patience=20, verbose=False)

# Loss history
# Loss history
loss_history = []

# Total population
N = data["population"].values[0]  # Assuming the population is constant and given in the data

def train_model(tensor_data, model, beta_net, optimizer, scheduler, early_stopping, num_epochs=1000):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        beta_net.train()

        running_loss = 0.0

        train_tensors, val_tensors = tensor_data["train"], tensor_data["val"]
        t_train, S_train, I_train, R_train, D_train, H_train, C_train = train_tensors

        optimizer.zero_grad()

        params = beta_net(t_train)
        model_output = model(t_train)

        loss = pinn_loss(tensor_data, params, model_output, t_train, N, device)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        scheduler.step()

        loss_history.append(running_loss)

        if early_stopping(running_loss):
            print(f"Early stopping at epoch {epoch}. No improvement in loss for {early_stopping.patience} epochs.")
            break

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {running_loss}")

    print("Finished Training")
    return loss_history

# Train the model
loss_history = train_model(tensor_data, model, beta_net, model_optimizer, model_scheduler, early_stopping, num_epochs=50000)

# Plot training and validation loss in log scale
plt.figure(figsize=(10, 5))
plt.plot(np.log10(loss_history), label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.title("Training and Validation Loss over Epochs (Log Scale)")
plt.legend()
plt.show()

