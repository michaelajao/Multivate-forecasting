import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from collections import deque
from sklearn.preprocessing import MinMaxScaler
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
data = load_and_preprocess_data("../../data/hos_data/merged_data.csv", areaname="South West", recovery_period=21, rolling_window=7, start_date="2020-04-01", end_date="2020-12-31")

data.head(10)

# def SEIRD_model(t, y, beta, gamma, mu, sigma, e, alpha, N):
#     """SEIRD model differential equations."""
#     S, E, I, R, D = y
#     dSdt = -beta * S * (e * E + I) / N
#     dEdt = beta * S * (e * E + I) / N - E / alpha
#     dIdt = E / alpha - (gamma + mu) * I
#     dRdt = gamma * I
#     dDdt = mu * I
#     return [dSdt, dEdt, dIdt, dRdt, dDdt]

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

# def scale_data(data, features):
#     """Scale the data using MinMaxScaler."""
#     scaler = MinMaxScaler()
#     scaled_data = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
#     return scaled_data, scaler  

# # Define features and data split
# features = ["S(t)", "active_cases", "recovered", "new_deceased"]

# # Scale the data
# scaled_data, scaler = scale_data(data, features)

# # Prepare tensors
# t_data, S_data, I_data, R_data, D_data = prepare_tensors(scaled_data, device)

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

class ModifiedTanh(nn.Module):
    def __init__(self, alpha, epsilon):
        super(ModifiedTanh, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, x):
        return 0.5 * torch.tanh(self.alpha * x) + self.epsilon

class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        identity = x.clone()
        out = self.fc(x)
        out = self.activation(out)
        if out.shape == identity.shape:
            out = out + identity
        return out

class StateNN(nn.Module):
    """Epidemiological network for predicting SEIRD model outputs."""
    def __init__(self, num_layers=4, hidden_neurons=20):
        super(StateNN, self).__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(ResBlock(hidden_neurons, hidden_neurons))
        layers.append(nn.Linear(hidden_neurons, 6)) # Adjust the output size to 6 (S, I, R, D, H, C)
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)

class ParamNN(nn.Module):
    """Neural network for predicting time-varying parameters."""
    def __init__(self, num_layers=4, hidden_neurons=20):
        super(ParamNN, self).__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(ResBlock(hidden_neurons, hidden_neurons))
        layers.append(nn.Linear(hidden_neurons, 8)) # Adjust the output size to 8 (beta, gamma, delta, rho, eta, kappa, mu, xi)
        self.net = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        params = torch.sigmoid(self.net(t))
        # Ensure beta, gamma, and mu are in a valid range
        beta = nn.Parameter(torch.sigmoid(params[:, 0]))
        gamma = nn.Parameter(torch.sigmoid(params[:, 1]))
        delta = nn.Parameter(torch.sigmoid(params[:, 2]))
        rho = nn.Parameter(torch.sigmoid(params[:, 3]))
        eta = nn.Parameter(torch.sigmoid(params[:, 4]))
        kappa = nn.Parameter(torch.sigmoid(params[:, 5]))
        mu = nn.Parameter(torch.sigmoid(params[:, 6]))
        xi = nn.Parameter(torch.sigmoid(params[:, 7]))
        return beta, gamma, delta, rho, eta, kappa, mu, xi
    

def network_prediction(t, model, device, scaler, N):
    """
    Generate predictions from the SEIRDNet model.

    Parameters:
    t (numpy array): Time input.
    model (SEIRDNet): Trained SEIRDNet model.
    device (torch.device): Device to run the model on.
    scaler (MinMaxScaler): Scaler used to normalize the data.
    N (float): Population size.

    Returns:
    np.array: Scaled predictions from the model.
    """
    # Convert time input to tensor and move to the appropriate device
    t_tensor = torch.from_numpy(t).float().view(-1, 1).to(device).requires_grad_(True)

    # Disable gradient computation for prediction
    with torch.no_grad():
        # Generate predictions from the model
        predictions = model(t_tensor)

        # Convert predictions to numpy array
        predictions = predictions.cpu().numpy()

        # Inverse transform the predictions to original scale
        predictions = scaler.inverse_transform(predictions)

    return predictions


# function for parameter estimation
def parameter_estimation(t, data, model, device, scaler, N):
    """
    Estimate parameters beta, gamma, and mu from the SEIRDNet model.

    Parameters:
    t (numpy array): Time input.
    data (dict): Dictionary containing the data tensors.
    model (SEIRDNet): Trained SEIRDNet model.
    device (torch.device): Device to run the model on.
    scaler (MinMaxScaler): Scaler used to normalize the data.
    N (float): Population size.

    Returns:
    np.array: Estimated parameters beta, gamma, and mu.
    """
    # Convert time input to tensor and move to the appropriate device
    t_tensor = torch.from_numpy(t).float().view(-1, 1).to(device).requires_grad_(True)

    # Disable gradient computation for prediction
    with torch.no_grad():
        # Generate predictions from the model
        predictions = model(t_tensor)

        # Convert predictions to numpy array
        predictions = predictions.cpu().numpy()

        # Inverse transform the predictions to original scale
        predictions = scaler.inverse_transform(predictions)

        # Extract the estimated parameters
        beta = predictions[:, 0]
        gamma = predictions[:, 1]
        mu = predictions[:, 2]

    return beta, gamma, mu


tensor_data, scaler = split_and_scale_data(data, train_size, features, device)


def pinn_loss(t, data, state_nn, param_nn, N, alpha, epsilon, train_size=None):
    """Physics-Informed Neural Network loss function."""
    
    # Predicted states
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
    
    
    # Compute gradients
    S_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    I_t = grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    R_t = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    D_t = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]
    H_t = grad(H_pred, t, grad_outputs=torch.ones_like(H_pred), create_graph=True)[0]
    C_t = grad(C_pred, t, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
    
    # Predicted parameters
    beta_pred, gamma_pred, delta_pred, rho_pred, eta_pred, kappa_pred, mu_pred, xi_pred = param_nn(t)
    
    # SEIRD model residuals
    e_tensor = torch.tensor(epsilon, dtype=torch.float32, device=device, requires_grad=True)
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device, requires_grad=True)
    
    e = torch.tanh(e_tensor)
    alpha = 2 * torch.tanh(alpha_tensor)
    
    dSdt, dIdt, dHdt, dCdt, dRdt, dDdt = SIHCRD_model(t, [S_pred, I_pred, H_pred, C_pred, R_pred, D_pred], beta_pred, gamma_pred, delta_pred, rho_pred, eta_pred, kappa_pred, mu_pred, xi_pred, N)
    
        # Generate a random subset of indices
    if train_size is not None:
        index = torch.randperm(train_size)
    else:
        index = torch.arange(len(t))
    
    # Compute data loss (MSE_u)
    data_loss = torch.mean((S_pred[index] - S_total[index]) ** 2) + torch.mean((I_pred[index] - I_total[index]) ** 2) + torch.mean((R_pred[index] - R_total[index]) ** 2) + torch.mean((D_pred[index] - D_total[index]) ** 2) + torch.mean((H_pred[index] - H_total[index]) ** 2) + torch.mean((C_pred[index] - C_total[index]) ** 2)
    
    # derivatives loss
    derivatives_loss = torch.mean((S_t - dSdt) ** 2) + torch.mean((I_t - dIdt) ** 2) + torch.mean((R_t - dRdt) ** 2) + torch.mean((D_t - dDdt) ** 2) + torch.mean((H_t - dHdt) ** 2) + torch.mean((C_t - dCdt) ** 2)
    
    # initial condition loss
    S0, I0, R0, D0, H0, C0 = S_pred[0], I_pred[0], R_pred[0], D_pred[0], H_pred[0], C_pred[0]
    initial_condition_loss = torch.mean((S0 - S_total[0]) ** 2) + torch.mean((I0 - I_total[0]) ** 2) + torch.mean((R0 - R_total[0]) ** 2) + torch.mean((D0 - D_total[0]) ** 2) + torch.mean((H0 - H_total[0]) ** 2) + torch.mean((C0 - C_total[0]) ** 2)
    
    # total loss
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
num_epochs = 50000
sigma = 1/5
N = data["population"].values[0]
alpha = 0.5
epsilon = 0.1

# Instantiate the neural networks with custom activation function
state_nn = StateNN(num_layers=6, hidden_neurons=32).to(device)
param_nn = ParamNN(num_layers=6, hidden_neurons=32).to(device)

# Optimizers
optimizer_state = optim.Adam(state_nn.parameters(), lr=learning_rate)
optimizer_param = optim.Adam(param_nn.parameters(), lr=learning_rate)

# Early stopping criteria
early_stopping = EarlyStopping(patience=20, verbose=False)

# Set the number of epochs for training
epochs = 100000

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
def train_model(epochs, t, data, state_nn, param_nn, optimizer_state, optimizer_param, N, sigma, alpha, epsilon, early_stopping, index):
    
    model_loss = []
    param_loss = []
    
    for epoch in tqdm(range(epochs)):
        state_nn.train()
        param_nn.train()
        
        # Zero gradients
        optimizer_state.zero_grad()
        optimizer_param.zero_grad()
        
        # Shuffle the training index for each epoch
        index = torch.randperm(len(data["train"][0]))
        
        # Get the model predictions
        loss = pinn_loss(t[index], data, state_nn, param_nn, N, alpha, epsilon, train_size=len(tensor_data["train"][0]))
        
        # Backward pass
        loss.backward()
        
        # Optimize
        optimizer_state.step()
        optimizer_param.step()
        
        model_loss.append(loss.item())
        param_loss.append(loss.item())
        
        
        if early_stopping(loss.item()):
            print(f"Early stopping at epoch {epoch}. No improvement in loss for {early_stopping.patience} epochs.")
            break            
    

        # Early stopping
        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")
            
        
    return model_loss, param_loss, state_nn, param_nn

# Train the model
model, param, state_nn, param_nn = train_model(epochs, t, tensor_data, state_nn, param_nn, optimizer_state, optimizer_param, N, sigma, alpha, epsilon, early_stopping, index)
    


# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(np.log10(state_nn), label='Model Loss')
plt.plot(np.log10(param_nn), label='Parameter Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



# # Predict and plot the results
# state_nn.eval()
# param_nn.eval()
# with torch.no_grad():
#     S_pred, E_pred, I_pred, R_pred, D_pred = state_nn(t_data).cpu().numpy().T
#     beta_pred, gamma_pred, mu_pred = param_nn(t_data)
#     beta_pred, gamma_pred, mu_pred = beta_pred.cpu().numpy(), gamma_pred.cpu().numpy(), mu_pred.cpu().numpy()
    
#     # plot the predictions 
#     fig, ax = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
    
#     ax[0].plot(S_pred, label='S(t) (Predicted)', color='blue')
#     ax[0].plot(S_data.cpu().numpy(), label='S(t) (Actual)', color='red', linestyle='dashed')
#     ax[0].set_ylabel('S(t)')
#     ax[0].legend()
    
#     ax[1].plot(I_pred, label='I(t) (Predicted)', color='blue')
#     ax[1].plot(I_data.cpu().numpy(), label='I(t) (Actual)', color='red', linestyle='dashed')
#     ax[1].set_ylabel('I(t)')
#     ax[1].legend()
    
#     ax[2].plot(R_pred, label='R(t) (Predicted)', color='blue')
#     ax[2].plot(R_data.cpu().numpy(), label='R(t) (Actual)', color='red', linestyle='dashed')
#     ax[2].set_ylabel('R(t)')
#     ax[2].legend()

#     ax[3].plot(D_pred, label='D(t) (Predicted)', color='blue')
#     ax[3].plot(D_data.cpu().numpy(), label='D(t) (Actual)', color='red', linestyle='dashed')
#     ax[3].set_ylabel('D(t)')
#     ax[3].legend()
    
#     plt.xlabel('Time (days)')
#     plt.tight_layout()
#     plt.show()
    
#     # plot the parameters
#     fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
#     ax[0].plot(beta_pred, label='beta(t) (Predicted)', color='blue')
#     ax[0].set_ylabel('beta(t)')
#     ax[0].legend()
    
#     ax[1].plot(gamma_pred, label='gamma(t) (Predicted)', color='blue')
#     ax[1].set_ylabel('gamma(t)')
#     ax[1].legend()
    
#     ax[2].plot(mu_pred, label='mu(t) (Predicted)', color='blue')
#     ax[2].set_ylabel('mu(t)')
#     ax[2].legend()
    
#     plt.xlabel('Time (days)')
#     plt.tight_layout()
#     plt.show()
    
    

    
    
    
    