import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm
from scipy.integrate import solve_ivp

# Set the random seed for reproducibility
plt.style.use("seaborn-v0_8-poster")
plt.rcParams.update({
    # "font.family": "sans-serif",
    # "font.sans-serif": ["Arial"],
    "font.size": 14,
    "text.usetex": False,
    "figure.figsize": [8, 5],
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
})

# Device setup for CUDA or CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
        # Print CUDA version
        print(f"CUDA version: {torch.version.cuda}")
        
        # List available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available. PyTorch will run on CPU.")
        
check_pytorch()

# SIHCRD model definition
def SIHCRD_model(t, y, beta, gamma, delta, alpha, N):
    S, I, H, C, R, D = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - (gamma + alpha) * I
    dHdt = alpha * I - (gamma + delta) * H
    dCdt = gamma * H - delta * C
    dRdt = gamma * (H + I)
    dDdt = delta * (H + C)
    return [dSdt, dIdt, dHdt, dCdt, dRdt, dDdt]

# Initial conditions function
def initial_conditions(N, I0, H0, C0, R0, D0):
    S0 = N - I0 - H0 - C0 - R0 - D0
    return [S0, I0, H0, C0, R0, D0]

# Solve the system of differential equations
def solve_SIHCRD_model(t0, tf, y0, beta, gamma, delta, alpha, N):
    sol = solve_ivp(SIHCRD_model, [t0, tf], y0, args=(beta, gamma, delta, alpha, N), dense_output=True)
    return sol

# Plotting function
def plot_solution(t, y, title):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(t, y[1], label="Infected", color="tab:blue")
    ax.plot(t, y[2], label="Hospitalized", color="tab:orange")
    ax.plot(t, y[3], label="Critical", color="tab:red")
    ax.plot(t, y[4], label="Recovered", color="tab:green")
    ax.plot(t, y[5], label="Deceased", color="tab:purple")
    ax.plot(t, y[0], label="Susceptible", color="tab:cyan")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Number of individuals")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"../../reports/figures/{title}.pdf")
    plt.show()

# Load and preprocess the data
def load_and_preprocess_data(filepath, recovery_period=16, rolling_window=7, start_date="2020-04-01"):
    df = pd.read_csv(filepath)
    required_columns = ["date", "cumulative_confirmed", "cumulative_deceased", "population", "covidOccupiedMVBeds", "hospitalCases"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["date"] = pd.to_datetime(df["date"])
    df["days_since_start"] = (df["date"] - pd.to_datetime(start_date)).dt.days

    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df["cumulative_deceased"].shift(recovery_period)
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)

    df["active_cases"] = df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]

    df["S(t)"] = df["population"] - df["active_cases"] - df["recovered"] - df["cumulative_deceased"]

    df = df[df["date"] >= pd.to_datetime(start_date)].reset_index(drop=True)
    df[["recovered", "active_cases", "S(t)"]] = df[["recovered", "active_cases", "S(t)"]].clip(lower=0)

    return df

# Split the data into training and validation sets
def split_data(data, train_start, train_end, val_start, val_end):
    t_mask = (data["date"] >= train_start) & (data["date"] <= train_end)
    train_data = data.loc[t_mask]

    v_mask = (data["date"] >= val_start) & (data["date"] <= val_end)
    val_data = data.loc[v_mask]

    return train_data, val_data

# Prepare tensors
def prepare_tensors(data, device):
    t = torch.tensor(range(1, len(data) + 1), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
    I = torch.tensor(data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
    R = torch.tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = torch.tensor(data["cumulative_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
    H = torch.tensor(data["hospitalCases"].values, dtype=torch.float32).view(-1, 1).to(device)
    C = torch.tensor(data["covidOccupiedMVBeds"].values, dtype=torch.float32).view(-1, 1).to(device)
    return t, I, R, D, H, C

# Define the neural network for epi-net
class EpiNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=5):
        super(EpiNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, output_size))
        self.net = nn.Sequential(*layers)
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

# Define the neural network for beta parameter estimation
class BetaNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10):
        super(BetaNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 1))
        self.net = nn.Sequential(*layers)
        self.init_xavier()

    def forward(self, t):
        return self.get_params(t)

    def init_xavier(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)
        self.net.apply(init_weights)

    def get_params(self, t):
        beta = torch.sigmoid(self.net(t)) * 0.9 + 0.1
        return beta

# Define the PINN loss function
def pinn_loss(tensor_data, beta_pred, model_output, t, N, device):
    I, H, C, R, D = tensor_data.unbind(1)
    S = N - I.sum() - H.sum() - C.sum() - R.sum() - D.sum()
    S_pred = S - beta_pred * S * I / N

    S = S.view(-1)
    I_pred, H_pred, C_pred, R_pred, D_pred = model_output.unbind(1)
    I_pred, H_pred, C_pred, R_pred, D_pred = (
        I_pred.view(-1),
        H_pred.view(-1),
        C_pred.view(-1),
        R_pred.view(-1),
        D_pred.view(-1),
    )

    s_t = torch.autograd.grad(outputs=S_pred, inputs=t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    i_t = torch.autograd.grad(outputs=I_pred, inputs=t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    h_t = torch.autograd.grad(outputs=H_pred, inputs=t, grad_outputs=torch.ones_like(H_pred), create_graph=True)[0]
    c_t = torch.autograd.grad(outputs=C_pred, inputs=t, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
    r_t = torch.autograd.grad(outputs=R_pred, inputs=t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    d_t = torch.autograd.grad(outputs=D_pred, inputs=t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]

    gamma = 0.1
    delta = 0.01
    alpha = 0.05

    dSdt = s_t + beta_pred * S * I / N
    dIdt = i_t - beta_pred * S * I / N - (gamma + alpha) * I
    dHdt = h_t - alpha * I - (gamma + delta) * H
    dCdt = c_t - gamma * H - delta * C
    dRdt = r_t - gamma * (H + I)
    dDdt = d_t - delta * (H + C)

    data_loss = torch.mean((I - I_pred) ** 2 + (H - H_pred) ** 2 + (C - C_pred) ** 2 + (R - R_pred) ** 2 + (D - D_pred) ** 2)
    physics_loss = torch.mean(dSdt**2 + dIdt**2 + dHdt**2 + dCdt**2 + dRdt**2 + dDdt**2)
    initial_condition_loss = torch.mean((I[0] - I_pred[0]) ** 2 + (H[0] - H_pred[0]) ** 2 + (C[0] - C_pred[0]) ** 2 + (R[0] - R_pred[0]) ** 2 + (D[0] - D_pred[0]) ** 2)

    loss = data_loss + physics_loss + initial_condition_loss
    return loss

# Early stopping class
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
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Training function
def train_model(model, beta_net, tensor_data, t_train, N, lr, num_epochs=1000, device=device, print_every=100, verbose=True):
    model_optimizer = optim.Adam(model.parameters(), lr=lr)
    params_optimizer = optim.Adam(beta_net.parameters(), lr=lr)
    model_scheduler = StepLR(model_optimizer, step_size=10000, gamma=0.9)
    params_scheduler = StepLR(params_optimizer, step_size=10000, gamma=0.9)
    early_stopping = EarlyStopping(patience=100, verbose=verbose)
    loss_history = []

    for epoch in tqdm(range(num_epochs)):
        model_optimizer.zero_grad()
        params_optimizer.zero_grad()

        model_output = model(t_train)
        beta_pred = beta_net(t_train)
        loss = pinn_loss(tensor_data, beta_pred, model_output, t_train, N, device)

        loss.backward()
        model_optimizer.step()
        params_optimizer.step()

        loss_history.append(loss.item())

        if verbose and (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item():.6f}")

        model_scheduler.step()
        params_scheduler.step()
        
        early_stopping(loss.item())
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model, beta_net, loss_history

# Generate predictions from the model
def network_prediction(model, t, N, device):
    t_tensor = torch.from_numpy(t).float().view(-1, 1).to(device)
    with torch.no_grad():
        predictions = model(t_tensor) * N
    return predictions.cpu().numpy()

# Plot results
def plot_results(t, I_data, R_data, D_data, model, title, N):
    model.eval()
    with torch.no_grad():
        predictions = model(t).cpu().numpy()

    t_np = t.cpu().detach().numpy().flatten()
    I_pred, H_pred, C_pred, R_pred, D_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3], predictions[:, 4]

    fig, axs = plt.subplots(6, 1, figsize=(10, 20))

    # Plotting S (Susceptible)
    S_pred = N - I_pred - H_pred - C_pred - R_pred - D_pred
    axs[0].plot(t_np, S_pred, 'r-', label='$S_{PINN}$')
    axs[0].set_title('S')
    axs[0].set_xlabel('Time t (days)')
    axs[0].legend()

    # Plotting I (Infected)
    axs[1].scatter(t_np, I_data.cpu().detach().numpy().flatten(), color='black', label='$I_{Data}$', s=10)
    axs[1].plot(t_np, I_pred, 'r-', label='$I_{PINN}$')
    axs[1].set_title('I')
    axs[1].set_xlabel('Time t (days)')
    axs[1].legend()

    # Plotting H (Hospitalized)
    axs[2].plot(t_np, H_pred, 'r-', label='$H_{PINN}$')
    axs[2].set_title('H')
    axs[2].set_xlabel('Time t (days)')
    axs[2].legend()

    # Plotting C (Critical)
    axs[3].plot(t_np, C_pred, 'r-', label='$C_{PINN}$')
    axs[3].set_title('C')
    axs[3].set_xlabel('Time t (days)')
    axs[3].legend()

    # Plotting R (Recovered)
    axs[4].scatter(t_np, R_data.cpu().detach().numpy().flatten(), color='black', label='$R_{Data}$', s=10)
    axs[4].plot(t_np, R_pred, 'r-', label='$R_{PINN}$')
    axs[4].set_title('R')
    axs[4].set_xlabel('Time t (days)')
    axs[4].legend()

    # Plotting D (Deceased)
    axs[5].scatter(t_np, D_data.cpu().detach().numpy().flatten(), color='black', label='$D_{Data}$', s=10)
    axs[5].plot(t_np, D_pred, 'r-', label='$D_{PINN}$')
    axs[5].set_title('D')
    axs[5].set_xlabel('Time t (days)')
    axs[5].legend()

    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{title}.pdf")
    plt.show()

# Plot training loss
def plot_loss(losses, title):
    plt.plot(np.arange(1, len(losses) + 1), losses, label='Loss', color='black')
    plt.yscale('log')
    plt.title(f"{title} loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.savefig(f"../../reports/figures/{title}_loss.pdf")
    plt.show()

# Plot time-varying beta parameter
def plot_beta(beta_net, t, title):
    beta_net.eval()
    with torch.no_grad():
        beta_pred = beta_net(t).cpu().numpy().flatten()

    t_np = t.cpu().detach().numpy().flatten()
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_np, beta_pred, label='Beta (time-varying parameter)', color='blue')
    plt.title('Time-varying Beta Parameter')
    plt.xlabel('Time t (days)')
    plt.ylabel('Beta')
    plt.legend()
    plt.savefig(f"../../reports/figures/{title}_beta.pdf")
    plt.show()

# Main script execution
if __name__ == "__main__":
    # Load and preprocess the data
    data = load_and_preprocess_data("../../data/processed/england_data.csv", recovery_period=21, rolling_window=7, start_date="2020-04-02")

    # Split the data into training and validation sets
    train_data, val_data = split_data(data, "2020-05-01", "2020-12-31", "2021-01-01", "2021-01-30")

    # Normalize the data using MinMaxScaler
    features = ["active_cases", "hospitalCases", "covidOccupiedMVBeds", "recovered", "cumulative_deceased"]
    scaler = MinMaxScaler()
    scaler.fit(train_data[features])
    scaled_train_data = pd.DataFrame(scaler.transform(train_data[features]), columns=features)
    scaled_val_data = pd.DataFrame(scaler.transform(val_data[features]), columns=features)

    # Prepare tensors for training and validation data
    t_train, I_train, R_train, D_train, H_train, C_train = prepare_tensors(scaled_train_data, device)
    t_val, I_val, R_val, D_val, H_val, C_val = prepare_tensors(scaled_val_data, device)
    train_tensor_data = torch.cat([I_train, H_train, C_train, R_train, D_train], dim=1)

    # Initialize models
    model = EpiNet(num_layers=5, hidden_neurons=32, output_size=5).to(device)
    beta_net = BetaNet(num_layers=5, hidden_neurons=32).to(device)

    # Train the model
    N = data["population"].values[0]
    model, beta_net, loss_history = train_model(model, beta_net, train_tensor_data, t_train, N, lr=1e-4, num_epochs=100000)

    # Plot loss history
    plot_loss(loss_history, "Training")

    # Evaluate the model on training data
    plot_results(t_train, I_train, R_train, D_train, model, "Training Data", N)

    # Predict on validation data
    plot_results(t_val, I_val, R_val, D_val, model, "Validation Data", N)

    # Plot time-varying beta parameter
    plot_beta(beta_net, t_train, "Training Data")
    plot_beta(beta_net, t_val, "Validation Data")
