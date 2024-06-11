import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.integrate import solve_ivp

from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm

# Set the random seed for reproducibility
plt.style.use("seaborn-v0_8-poster")
plt.rcParams.update(
    {
        "font.size": 14,
        "text.usetex": False,
        "figure.figsize": [8, 4],
        "figure.facecolor": "white",
        "figure.autolayout": True,
        "figure.dpi": 400,
        "savefig.dpi": 400,
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

# Device setup for CUDA or CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_pytorch():
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

def SIHCRD_model(t, y, beta, gamma, delta, rho, eta, kappa, mu, xi, N):
    S, I, H, C, R, D = y
    dSdt = -(beta * I / N) * S
    dIdt = (beta * S / N) * I - (gamma + rho + delta) * I
    dHdt = rho * I - (eta + kappa) * H
    dCdt = eta * H - (mu + xi) * C
    dRdt = gamma * I + kappa * H + mu * C
    dDdt = delta * I + xi * C
    return [dSdt, dIdt, dHdt, dCdt, dRdt, dDdt]

def initial_conditions(N, I0, H0, C0, R0, D0):
    S0 = N - I0 - H0 - C0 - R0 - D0
    return [S0, I0, H0, C0, R0, D0]

t0, tf = 0, 360
N = 1e6
I0, H0, C0, R0, D0 = 100, 10, 5, 1, 1
beta, gamma, delta, rho, eta, kappa, mu, xi = 0.3, 0.1, 0.01, 0.05, 0.05, 0.05, 0.05, 0.01

y0 = initial_conditions(N, I0, H0, C0, R0, D0)

sol = solve_ivp(
    SIHCRD_model,
    [t0, tf],
    y0,
    args=(beta, gamma, delta, rho, eta, kappa, mu, xi, N),
    dense_output=True,
)

t = np.linspace(t0, tf, 1000)
y = sol.sol(t)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, y[1], label="Infected", color="tab:blue")
ax.plot(t, y[2], label="Hospitalized", color="tab:orange")
ax.plot(t, y[3], label="Critical", color="tab:red")
ax.plot(t, y[4], label="Recovered", color="tab:green")
ax.plot(t, y[5], label="Deceased", color="tab:purple")
ax.plot(t, y[0], label="Susceptible", color="tab:cyan")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Number of individuals")
ax.set_title("SIHCRD Model")
ax.legend()
plt.show()

df = pd.read_csv("../../data/processed/england_data.csv")

def load_and_preprocess_data(filepath, recovery_period=16, rolling_window=7, start_date="2020-04-01"):
    df = pd.read_csv(filepath)
    required_columns = [
        "date",
        "cumulative_confirmed",
        "cumulative_deceased",
        "population",
        "covidOccupiedMVBeds",
        "hospitalCases",
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["date"] = pd.to_datetime(df["date"])
    df["days_since_start"] = (df["date"] - pd.to_datetime(start_date)).dt.days

    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df["cumulative_deceased"].shift(recovery_period)
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)

    df["active_cases"] = df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"].fillna(0).clip(lower=0)
    
        # Calculate susceptible cases
    df["susceptible"] = df["population"] - (
        df["recovered"] + df["cumulative_deceased"] + df["active_cases"]
    ).clip(lower=0)

    for col in [
        "cumulative_confirmed",
        "cumulative_deceased",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "recovered",
        "active_cases",
        "susceptible"
    ]:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    df = df[df["date"] >= pd.to_datetime(start_date)].reset_index(drop=True)

    return df

data = load_and_preprocess_data(
    "../../data/processed/england_data.csv",
    recovery_period=21,
    start_date="2020-04-01",
)

train_data_start = "2020-05-01"
train_data_end = "2020-10-31"
val_data_start = "2020-11-01"
val_data_end = "2020-12-31"

t_mask = (data["date"] >= train_data_start) & (data["date"] <= train_data_end)
train_data = data.loc[t_mask]

v_mask = (data["date"] >= val_data_start) & (data["date"] <= val_data_end)
val_data = data.loc[v_mask]

def prepare_tensors(data, device):
    t = torch.tensor(range(1, len(data) + 1), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
    S = torch.tensor(data["susceptible"].values, dtype=torch.float32).view(-1, 1).to(device)
    I = torch.tensor(data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
    H = torch.tensor(data["hospitalCases"].values, dtype=torch.float32).view(-1, 1).to(device)
    C = torch.tensor(data["covidOccupiedMVBeds"].values, dtype=torch.float32).view(-1, 1).to(device)
    R = torch.tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = torch.tensor(data["cumulative_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
    return t, S, I, H, C, R, D

features = ["susceptible", "active_cases", "hospitalCases", "covidOccupiedMVBeds", "recovered",  "cumulative_deceased"]
scaler = MinMaxScaler()
scaler.fit(train_data[features])

scaled_train_data = pd.DataFrame(scaler.transform(train_data[features]), columns=features)
scaled_val_data = pd.DataFrame(scaler.transform(val_data[features]), columns=features)

t_train, S_train, I_train, H_train, C_train, R_train, D_train = prepare_tensors(scaled_train_data, device)
t_val, S_val, I_val, H_val, C_val, R_val, D_val = prepare_tensors(scaled_val_data, device)

train_tensor_data = torch.cat([S_train, I_train, H_train, C_train, R_train, D_train], dim=1)
val_tensor_data = torch.cat([S_val, I_val, H_val, C_val, R_val, D_val], dim=1)


plt.plot(I_train.cpu().detach().numpy(), label="Active Cases")
plt.title("Active Cases in England")
plt.xlabel("Days since start")
plt.ylabel("Active Cases")
plt.legend()
plt.show()

plt.plot(I_train.cpu().detach().numpy(), label="Training Data")
plt.plot(
    range(len(I_train), len(I_train) + len(I_val)),
    I_val.cpu().detach().numpy(),
    label="Validation Data",
)
plt.axvline(x=len(I_train), color="black", linestyle="--", label="Split")
plt.title("Training and Validation Data Split")
plt.xlabel("Days since start")
plt.ylabel("Active Cases")
plt.legend()
plt.show()

class EpiNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=6):
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
            output = torch.sigmoid(self.net(t))
            return output

    # def forward(self, t):
    #     output = self.net(t)
    #     return torch.relu(output)  # Ensure non-negative outputs

    def init_xavier(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

        self.net.apply(init_weights)


class BetaNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10):
        super(BetaNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])

        layers.append(nn.Linear(hidden_neurons, 8))
        self.net = nn.Sequential(*layers)
        self.init_xavier()

    def forward(self, t):
        return self.net(t)

    def get_params(self, t):
        raw_params = self.forward(t)
        # Apply non-negative constraints to the parameters using sigmoid
        beta = torch.sigmoid(raw_params[:, 0]) 
        gamma = torch.sigmoid(raw_params[:, 1]) 
        delta = torch.sigmoid(raw_params[:, 2])
        rho = torch.sigmoid(raw_params[:, 3])
        eta = torch.sigmoid(raw_params[:, 4])
        kappa = torch.sigmoid(raw_params[:, 5])
        mu = torch.sigmoid(raw_params[:, 6])
        xi = torch.sigmoid(raw_params[:, 7])
        return torch.stack([beta, gamma, delta, rho, eta, kappa, mu, xi], dim=1)

    def init_xavier(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)
        self.net.apply(init_weights)

    

    # def init_xavier(self):
    #     def init_weights(layer):
    #         if isinstance(layer, nn.Linear):
    #             g = nn.init.calculate_gain("tanh")
    #             nn.init.xavier_normal_(layer.weight, gain=g)
    #             if layer.bias is not None:
    #                 layer.bias.data.fill_(0.01)``

    #     self.net.apply(init_weights)



def pinn_loss(tensor_data, parameters, model_output, t, N, device):
    S, I, H, C, R, D = tensor_data.unbind(1)

    beta_pred, gamma_pred, delta_pred, rho_pred, eta_pred, kappa_pred, mu_pred, xi_pred = parameters[:, 0], parameters[:, 1], parameters[:, 2], parameters[:, 3], parameters[:, 4], parameters[:, 5], parameters[:, 6], parameters[:, 7]

    N = N / N

    S_pred, I_pred, H_pred, C_pred, R_pred, D_pred = model_output[:, 0], model_output[:, 1], model_output[:, 2], model_output[:, 3], model_output[:, 4], model_output[:, 5]
    
    
    # Compute the gradients
    s_t = grad(outputs=S_pred, inputs=t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    i_t = grad(outputs=I_pred, inputs=t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    h_t = grad(outputs=H_pred, inputs=t, grad_outputs=torch.ones_like(H_pred), create_graph=True)[0]
    c_t = grad(outputs=C_pred, inputs=t, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
    r_t = grad(outputs=R_pred, inputs=t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    d_t = grad(outputs=D_pred, inputs=t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]
    
    # Compute the derivatives
    dSdt = -beta_pred * S_pred * I_pred / N
    dIdt = beta_pred * S_pred * I_pred / N - (gamma_pred + rho_pred + delta_pred) * I_pred
    dHdt = rho_pred * I_pred - (eta_pred + kappa_pred) * H_pred
    dCdt = eta_pred * H_pred - (mu_pred + xi_pred) * C_pred
    dRdt = gamma_pred * I_pred + kappa_pred * H_pred + mu_pred * C_pred
    dDdt = delta_pred * I_pred + xi_pred * C_pred
    

    data_loss = torch.mean((S - S_pred) ** 2 + (I - I_pred) ** 2 + (H - H_pred) ** 2 + (C - C_pred) ** 2 + (R - R_pred) ** 2 + (D - D_pred) ** 2)
    physics_loss = torch.mean((s_t - dSdt) ** 2 + (i_t - dIdt) ** 2 + (h_t - dHdt) ** 2 + (c_t - dCdt) ** 2 + (r_t - dRdt) ** 2 + (d_t - dDdt) ** 2)
    initial_condition_loss = torch.mean((S[0] - S_pred[0]) ** 2 + (I[0] - I_pred[0]) ** 2 + (H[0] - H_pred[0]) ** 2 + (C[0] - C_pred[0]) ** 2 + (R[0] - R_pred[0]) ** 2 + (D[0] - D_pred[0]) ** 2)
    
    loss = data_loss + physics_loss + initial_condition_loss
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
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

model = EpiNet(num_layers=5, hidden_neurons=32, output_size=6).to(device)
beta_net = BetaNet(num_layers=1, hidden_neurons=32).to(device)

def train_model(
    model,
    beta_net,
    tensor_data,
    t_train,
    N,
    lr,
    num_epochs=1000,
    device=device,
    print_every=100,
    verbose=True,
):
    # Define the optimizers with L2 regularization
    optimizer =optim.AdamW(list(model.parameters()) + list(beta_net.parameters()), lr=lr, weight_decay=1e-2)
    
    # model_optimizer = optim.Adam(model.parameters(), lr=lr)
    # params_optimizer = optim.Adam(beta_net.parameters(), lr=lr)

    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.9)
    # model_scheduler = StepLR(model_optimizer, step_size=5000, gamma=0.9)
    # params_scheduler = StepLR(params_optimizer, step_size=5000, gamma=0.9)

    # Initialize the early stopping object
    early_stopping = EarlyStopping(patience=100, verbose=verbose)

    # Initialize the loss history
    loss_history = []

    # Train the model
    for epoch in tqdm(range(num_epochs)):
        model.train()
        beta_net.train()
        
        optimizer.zero_grad()
        # model_optimizer.zero_grad()
        # params_optimizer.zero_grad()

        # Forward pass to compute the predicted values on the training data
        model_output = model(t_train)

        # Forward pass to compute the predicted parameters
        parameters = beta_net.get_params(t_train)

        # Compute the loss
        loss = pinn_loss(tensor_data, parameters, model_output, t_train, N, device)

        # Backward pass
        loss.backward()

        # # Update the parameters
        # model_optimizer.step()
        # params_optimizer.step()
        optimizer.step()

        # Append the loss to the loss history
        loss_history.append(loss.item())

        # Print the loss every `print_every` epochs
        if (epoch + 1) % print_every == 0 or epoch == 0:
            if verbose:
                print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item():.6f}")

        # # Step the learning rate scheduler
        # model_scheduler.step()
        # params_scheduler.step()
        scheduler.step()
        
        # Check for early stopping
        early_stopping(loss.item())
        if early_stopping.early_stop:
            if verbose:
                print("Early stopping")
            break

    return model_output, parameters, loss_history


N = data["population"].values[0]

model_output, parameters, loss_history = train_model(
    model,
    beta_net,
    train_tensor_data,
    t_train,
    N,
    lr=1e-4,
    num_epochs=20000,
    device=device,
    print_every=500,
    verbose=True,
)

# Plot the loss history in base 10
plt.plot(np.log10(loss_history))
plt.title("Log Loss History")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.show()

# Plot the loss history
plt.plot(np.arange(1, len(loss_history) + 1), loss_history, label='Loss', color='black')
plt.yscale('log')
# plt.title(f"{title} loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.legend()
plt.show()

# Plot the actual and predicted of the training data for active cases
model_output = model(t_train)
plt.plot(I_train.cpu().detach().numpy(), label="Actual Active Cases")
plt.plot(model_output[:, 1].cpu().detach().numpy(), label="Predicted Active Cases")
plt.title("Actual vs Predicted Active Cases")
plt.xlabel("Days since start")
plt.ylabel("Active Cases")
plt.legend()
plt.show()


def plot_training_results(t_train, actual_data, model_output):
    t_train_np = t_train.cpu().detach().numpy().flatten()
    actual_data_np = actual_data.cpu().detach().numpy().flatten()
    predicted_data_np = model_output[:, 1].cpu().detach().numpy().flatten()

    plt.figure(figsize=(10, 5))
    plt.plot(t_train_np, actual_data_np, label='Actual Infected', color='blue')
    plt.plot(t_train_np, predicted_data_np, label='Predicted Infected', color='red', linestyle='--')
    plt.title('Comparison of Actual and Predicted Infected Cases')
    plt.xlabel('Time (days since start)')
    plt.ylabel('Number of Infected Individuals')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_training_results(t_train, I_train, model_output)

# Predict parameters and plot them
t_train_np = t_train.cpu().detach().numpy().flatten()
beta_values, gamma_values, delta_values, rho_values, eta_values, kappa_values, mu_values, xi_values = parameters.cpu().detach().numpy().T

fig, axs = plt.subplots(4, 2, figsize=(12, 12))
axs[0, 0].plot(t_train_np, beta_values, label="β")
axs[0, 0].set_title("β")
axs[0, 0].set_xlabel("Days since start")
axs[0, 0].set_ylabel("Value")
axs[0, 0].legend()

axs[0, 1].plot(t_train_np, gamma_values, label="γ")
axs[0, 1].set_title("γ")
axs[0, 1].set_xlabel("Days since start")
axs[0, 1].set_ylabel("Value")
axs[0, 1].legend()

axs[1, 0].plot(t_train_np, delta_values, label="δ")
axs[1, 0].set_title("δ")
axs[1, 0].set_xlabel("Days since start")
axs[1, 0].set_ylabel("Value")
axs[1, 0].legend()

axs[1, 1].plot(t_train_np, rho_values, label="ρ")
axs[1, 1].set_title("ρ")
axs[1, 1].set_xlabel("Days since start")
axs[1, 1].set_ylabel("Value")
axs[1, 1].legend()

axs[2, 0].plot(t_train_np, eta_values, label="η")
axs[2, 0].set_title("η")
axs[2, 0].set_xlabel("Days since start")
axs[2, 0].set_ylabel("Value")
axs[2, 0].legend()

axs[2, 1].plot(t_train_np, kappa_values, label="κ")
axs[2, 1].set_title("κ")
axs[2, 1].set_xlabel("Days since start")
axs[2, 1].set_ylabel("Value")
axs[2, 1].legend()

axs[3, 0].plot(t_train_np, mu_values, label="μ")
axs[3, 0].set_title("μ")
axs[3, 0].set_xlabel("Days since start")
axs[3, 0].set_ylabel("Value")
axs[3, 0].legend()

axs[3, 1].plot(t_train_np, xi_values, label="ξ")
axs[3, 1].set_title("ξ")
axs[3, 1].set_xlabel("Days since start")
axs[3, 1].set_ylabel("Value")
axs[3, 1].legend()

plt.tight_layout()
plt.show()

def plot_model_output(t_train, model_output):
    t_train_np = t_train.cpu().detach().numpy().flatten()
    S_pred, I_pred, H_pred, C_pred, R_pred, D_pred = model_output.cpu().detach().numpy().T

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    axs[0, 0].plot(t_train_np, S_pred, label="Susceptible")
    axs[0, 0].set_title("Susceptible")
    axs[0, 0].set_xlabel("Days since start")
    axs[0, 0].set_ylabel("Value")
    axs[0, 0].legend()

    axs[0, 1].plot(t_train_np, I_pred, label="Infected")
    axs[0, 1].set_title("Infected")
    axs[0, 1].set_xlabel("Days since start")
    axs[0, 1].set_ylabel("Value")
    axs[0, 1].legend()

    axs[1, 0].plot(t_train_np, H_pred, label="Hospitalized")
    axs[1, 0].set_title("Hospitalized")
    axs[1, 0].set_xlabel("Days since start")
    axs[1, 0].set_ylabel("Value")
    axs[1, 0].legend()

    axs[1, 1].plot(t_train_np, C_pred, label="Critical")
    axs[1, 1].set_title("Critical")
    axs[1, 1].set_xlabel("Days since start")
    axs[1, 1].set_ylabel("Value")
    axs[1, 1].legend()

    axs[2, 0].plot(t_train_np, R_pred, label="Recovered")
    axs[2, 0].set_title("Recovered")
    axs[2, 0].set_xlabel("Days since start")
    axs[2, 0].set_ylabel("Value")
    axs[2, 0].legend()

    axs[2, 1].plot(t_train_np, D_pred, label="Deceased")
    axs[2, 1].set_title("Deceased")
    axs[2, 1].set_xlabel("Days since start")
    axs[2, 1].set_ylabel("Value")
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()

plot_model_output(t_train, model_output)

# get the parameters predicted by the model
parameters = beta_net.get_params(t_train)

# Extract the parameter values
beta_values, gamma_values, delta_values, rho_values, eta_values, kappa_values, mu_values, xi_values = parameters.cpu().detach().numpy().T

fig, axs = plt.subplots(4, 2, figsize=(12, 12))
axs[0, 0].plot(t_train.cpu().detach().numpy(), beta_values, label="β")
axs[0, 0].set_title("β")
axs[0, 0].set_xlabel("Days since start")
axs[0, 0].set_ylabel("Value")
axs[0, 0].legend()

axs[0, 1].plot(t_train.cpu().detach().numpy(), gamma_values, label="γ")
axs[0, 1].set_title("γ")
axs[0, 1].set_xlabel("Days since start")
axs[0, 1].set_ylabel("Value")
axs[0, 1].legend()

axs[1, 0].plot(t_train.cpu().detach().numpy(), delta_values, label="δ")
axs[1, 0].set_title("δ")
axs[1, 0].set_xlabel("Days since start")
axs[1, 0].set_ylabel("Value")
axs[1, 0].legend()

axs[1, 1].plot(t_train.cpu().detach().numpy(), rho_values, label="ρ")
axs[1, 1].set_title("ρ")
axs[1, 1].set_xlabel("Days since start")
axs[1, 1].set_ylabel("Value")
axs[1, 1].legend()

axs[2, 0].plot(t_train.cpu().detach().numpy(), eta_values, label="η")
axs[2, 0].set_title("η")
axs[2, 0].set_xlabel("Days since start")
axs[2, 0].set_ylabel("Value")
axs[2, 0].legend()

axs[2, 1].plot(t_train.cpu().detach().numpy(), kappa_values, label="κ")
axs[2, 1].set_title("κ")
axs[2, 1].set_xlabel("Days since start")
axs[2, 1].set_ylabel("Value")
axs[2, 1].legend()

axs[3, 0].plot(t_train.cpu().detach().numpy(), mu_values, label="μ")
axs[3, 0].set_title("μ")
axs[3, 0].set_xlabel("Days since start")
axs[3, 0].set_ylabel("Value")
axs[3, 0].legend()

axs[3, 1].plot(t_train.cpu().detach().numpy(), xi_values, label="ξ")
axs[3, 1].set_title("ξ")
axs[3, 1].set_xlabel("Days since start")
axs[3, 1].set_ylabel("Value")
axs[3, 1].legend()

plt.tight_layout()
plt.show()


