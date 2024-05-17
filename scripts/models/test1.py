import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from scipy.integrate import solve_ivp

# Ensure the folders exist
os.makedirs("../../models", exist_ok=True)
os.makedirs("../../reports/figures", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the default style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "text.usetex": False,
    "figure.figsize": (8, 5),
    "figure.facecolor": "white",
    "figure.autolayout": True,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "axes.labelsize": 14,
    "axes.titlesize": 20,
    "axes.facecolor": "white",
    "legend.fontsize": 12,
    "legend.frameon": False,
    "legend.loc": "best",
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(filepath, recovery_period=21, rolling_window=7, start_date="2020-04-01"):
    df = pd.read_csv(filepath)
    required_columns = [
        "date", "cumulative_confirmed", "cumulative_deceased",
        "population", "new_confirmed", "new_deceased",
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["date"] = pd.to_datetime(df["date"])
    df["days_since_start"] = (df["date"] - pd.to_datetime(start_date)).dt.days

    for col in ["new_confirmed", "new_deceased", "cumulative_confirmed", "cumulative_deceased"]:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df["cumulative_deceased"].shift(recovery_period)
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)

    df["active_cases"] = df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]

    df = df[df["date"] >= pd.to_datetime(start_date)].reset_index(drop=True)
    df[["recovered", "active_cases"]] = df[["recovered", "active_cases"]].clip(lower=0)

    return df

def get_region_name_from_filepath(filepath):
    base = os.path.basename(filepath)
    return os.path.splitext(base)[0]

path = "../../data/region_daily_data/Yorkshire and the Humber.csv"
region_name = get_region_name_from_filepath(path)
df = load_and_preprocess_data(f"../../data/region_daily_data/{region_name}.csv")

start_date = "2020-04-01"
end_date = "2020-08-31"
mask = (df["date"] >= start_date) & (df["date"] <= end_date)
training_data = df.loc[mask]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
columns_to_scale = ["active_cases", "recovered", "cumulative_deceased"]
training_data[columns_to_scale] = scaler.fit_transform(training_data[columns_to_scale])

# Convert columns to tensors
t_data = torch.tensor(range(len(training_data)), dtype=torch.float32).view(-1, 1).requires_grad_(True).to(device)
I_data = torch.tensor(training_data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
R_data = torch.tensor(training_data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
D_data = torch.tensor(training_data["cumulative_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
SIRD_tensor = torch.cat([I_data, R_data, D_data], dim=1).to(device)

class SEIRNet(nn.Module):
    def __init__(self, inverse=False, init_beta=None, init_gamma=None, init_delta=None, retrain_seed=42, num_layers=4, hidden_neurons=20):
        super(SEIRNet, self).__init__()
        self.retrain_seed = retrain_seed
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 4))
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
        torch.manual_seed(self.retrain_seed)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(init_weights)

def seird_loss(model, model_output, SIRD_tensor, t_tensor, N, sigma=1/5, beta=None, gamma=None, delta=None):
    I_pred, R_pred, D_pred, E_pred = model_output[:, 0], model_output[:, 1], model_output[:, 2], model_output[:, 3]
    S_pred = N - I_pred - R_pred - D_pred - E_pred

    I_t = torch.autograd.grad(I_pred, t_tensor, torch.ones_like(I_pred), create_graph=True)[0]
    R_t = torch.autograd.grad(R_pred, t_tensor, torch.ones_like(R_pred), create_graph=True)[0]
    D_t = torch.autograd.grad(D_pred, t_tensor, torch.ones_like(D_pred), create_graph=True)[0]
    E_t = torch.autograd.grad(E_pred, t_tensor, torch.ones_like(E_pred), create_graph=True)[0]

    if beta is None:
        beta, gamma, delta = model.beta, model.gamma, model.delta

    dSdt = -(beta * S_pred * I_pred) / N
    dEdt = (beta * S_pred * I_pred) / N - sigma * E_pred
    dIdt = sigma * E_pred - gamma * I_pred - delta * I_pred
    dRdt = gamma * I_pred
    dDdt = delta * I_pred

    loss = torch.mean((I_t - dIdt) ** 2) + torch.mean((R_t - dRdt) ** 2) + torch.mean((D_t - dDdt) ** 2) + torch.mean((E_t - dEdt) ** 2)
    loss += torch.mean((model_output[:, :3] - SIRD_tensor) ** 2)  # Data fitting loss for I, R, D
    return loss

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

def train(model, t_tensor, SIRD_tensor, epochs=1000, lr=0.001, N=None, sigma=1/5, beta=None, gamma=None, delta=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.9)
    early_stopping = EarlyStopping(patience=100, verbose=True)

    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        shuffled_indices = torch.randperm(len(t_tensor))
        t_shuffled = t_tensor[shuffled_indices]
        SIRD_shuffled = SIRD_tensor[shuffled_indices]

        model_output = model(t_shuffled)

        loss = seird_loss(model, model_output, SIRD_shuffled, t_shuffled, N, sigma, beta, gamma, delta)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if epoch % 1000 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

        early_stopping(loss)
        if early_stopping.early_stop:
            print("Early stopping")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
            }, f"../../models/{model.__class__.__name__}.pt")
            print("Model saved")
            break

    print("Training finished")
    return losses

def plot_results(t, I_data, R_data, D_data, model, title, N):
    model.eval()
    with torch.no_grad():
        predictions = model(t).cpu().numpy()

    t_np = t.cpu().detach().numpy().flatten()
    I_pred, R_pred, D_pred, E_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    S_pred = N - I_pred - R_pred - D_pred - E_pred

    fig, axs = plt.subplots(1, 5, figsize=(30, 6))

    # Plotting I (Infected)
    axs[0].scatter(t_np, I_data.cpu().detach().numpy().flatten(), color='black', label='$I_{Data}$', s=10)
    axs[0].plot(t_np, I_pred, 'r-', label='$I_{PINN}$')
    axs[0].set_title('I')
    axs[0].set_xlabel('Time t (days)')
    axs[0].set_ylabel('Number of individuals')
    axs[0].legend()

    # Plotting S (Susceptible)
    axs[1].plot(t_np, S_pred, 'r-', label='$S_{PINN}$')
    axs[1].set_title('S')
    axs[1].set_xlabel('Time t (days)')
    axs[1].legend()

    # Plotting E (Exposed)
    axs[2].plot(t_np, E_pred, 'r-', label='$E_{PINN}$')
    axs[2].set_title('E')
    axs[2].set_xlabel('Time t (days)')
    axs[2].legend()

    # Plotting R (Recovered)
    axs[3].scatter(t_np, R_data.cpu().detach().numpy().flatten(), color='black', label='$R_{Data}$', s=10)
    axs[3].plot(t_np, R_pred, 'r-', label='$R_{PINN}$')
    axs[3].set_title('R')
    axs[3].set_xlabel('Time t (days)')
    axs[3].legend()

    # Plotting D (Deceased)
    axs[4].scatter(t_np, D_data.cpu().detach().numpy().flatten(), color='black', label='$D_{Data}$', s=10)
    axs[4].plot(t_np, D_pred, 'r-', label='$D_{PINN}$')
    axs[4].set_title('D')
    axs[4].set_xlabel('Time t (days)')
    axs[4].legend()

    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{title}.pdf")
    plt.show()

# Plot the log10 of the loss
def plot_loss(losses, title):
    plt.plot(np.log10(losses))
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Log10 Loss")
    plt.show()

model_forward = SEIRNet(num_layers=5, hidden_neurons=32)
model_forward.to(device)
losses = train(model_forward, t_data, SIRD_tensor, epochs=50000, lr=0.001, N=N, sigma=1/5, beta=0.2, gamma=0.05, delta=0.01)

plot_results(t_data, I_data, R_data, D_data, model_forward, "Forward Model Results", N)
plot_loss(losses, "Forward Model Loss")

model_inverse = SEIRNet(inverse=True, init_beta=0.1, init_gamma=0.01, init_delta=0.01, num_layers=5, hidden_neurons=32)
model_inverse.to(device)
losses = train(model_inverse, t_data, SIRD_tensor, epochs=50000, lr=0.001, N=N, sigma=1/5)

plot_results(t_data, I_data, R_data, D_data, model_inverse, "Inverse Model Results", N)
plot_loss(losses, "Inverse Model Loss")

def extract_parameters(model):
    try:
        beta_predicted = model.beta.item()
        gamma_predicted = model.gamma.item()
        delta_predicted = model.delta.item()
        return beta_predicted, gamma_predicted, delta_predicted
    except AttributeError:
        print("Model does not have the requested parameters.")
        return None, None, None

beta_predicted, gamma_predicted, delta_predicted = extract_parameters(model_inverse)

def solve_seird_ode(beta, gamma, delta, N, I0, R0, D0, E0, t):
    S0 = N - I0 - R0 - D0 - E0
    sigma = 1/5
    def seird_model(t, y):
        S, I, R, D, E = y
        dSdt = -(beta * S * I) / N
        dEdt = (beta * S * I) / N - sigma * E
        dIdt = sigma * E - gamma * I - delta * I
        dRdt = gamma * I
        dDdt = delta * I
        return [dSdt, dEdt, dIdt, dRdt, dDdt]

    y0 = [S0, I0, R0, D0, E0]
    sol = solve_ivp(seird_model, [t[0], t[-1]], y0, t_eval=t, vectorized=True)
    return sol.y

# Get the final conditions and parameters from the inverse model
I0 = I_data[-1].item()
R0 = R_data[-1].item()
D0 = D_data[-1].item()
E0 = (beta_predicted * I0) / (1/5)  # Initial guess for E0

t_np = t_data.cpu().detach().numpy().flatten()
sol = solve_seird_ode(beta_predicted, gamma_predicted, delta_predicted, 1.0, I0, R0, D0, E0, t_np)

# Plot the ODE solution
def plot_ode_solution(t, sol, title):
    S, E, I, R, D = sol
    fig, axs = plt.subplots(1, 5, figsize=(30, 6))

    axs[0].plot(t, I, 'r-', label='$I_{ODE}$')
    axs[0].set_title('I')
    axs[0].set_xlabel('Time t (days)')
    axs[0].set_ylabel('Number of individuals')
    axs[0].legend()

    axs[1].plot(t, S, 'r-', label='$S_{ODE}$')
    axs[1].set_title('S')
    axs[1].set_xlabel('Time t (days)')
    axs[1].legend()

    axs[2].plot(t, E, 'r-', label='$E_{ODE}$')
    axs[2].set_title('E')
    axs[2].set_xlabel('Time t (days)')
    axs[2].legend()

    axs[3].plot(t, R, 'r-', label='$R_{ODE}$')
    axs[3].set_title('R')
    axs[3].set_xlabel('Time t (days)')
    axs[3].legend()

    axs[4].plot(t, D, 'r-', label='$D_{ODE}$')
    axs[4].set_title('D')
    axs[4].set_xlabel('Time t (days)')
    axs[4].legend()

    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{title}.pdf")
    plt.show()

plot_ode_solution(t_np, sol, "SEIRD ODE Solution")
