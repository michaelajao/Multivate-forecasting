import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from torch import tensor
from scipy import integrate

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


def load_and_preprocess_data(
    filepath,
    areaname,
    recovery_period=16,
    rolling_window=7,
    start_date="2020-04-01",
    end_date="2020-12-31",
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
    df["S(t)"] = (
        df["population"]
        - df["active_cases"]
        - df["recovered"]
        - df["cumulative_deceased"]
    )

    cols_to_smooth = [
        "S(t)",
        "cumulative_confirmed",
        "cumulative_deceased",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "recovered",
        "active_cases",
    ]
    for col in cols_to_smooth:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    return df


# Load and preprocess the data
data = load_and_preprocess_data(
    "../../data/hos_data/merged_data.csv",
    areaname="South West",
    recovery_period=21,
    start_date="2020-04-01",
    end_date="2020-12-31",
)


class SIRDNet(nn.Module):
    """Epidemiological network for predicting SIRD model outputs."""

    def __init__(
        self,
        inverse=False,
        init_beta=None,
        init_gamma=None,
        init_mu=None,
        retain_seed=42,
        num_layers=4,
        hidden_neurons=20,
    ):
        super(SIRDNet, self).__init__()
        self.retain_seed = retain_seed
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 4))  # Output size is 4 for SIRD model
        self.net = nn.Sequential(*layers)

        if inverse:
            self._beta = nn.Parameter(
                torch.tensor(
                    [init_beta if init_beta is not None else torch.rand(1)],
                    device=device,
                ),
                requires_grad=True,
            )
            self._gamma = nn.Parameter(
                torch.tensor(
                    [init_gamma if init_gamma is not None else torch.rand(1)],
                    device=device,
                ),
                requires_grad=True,
            )
            self._mu = nn.Parameter(
                torch.tensor(
                    [init_mu if init_mu is not None else torch.rand(1)], device=device
                ),
                requires_grad=True,
            )
        else:
            self._beta = None
            self._gamma = None
            self._mu = None

        self.init_xavier()

    def forward(self, t):
        return self.net(t)

    @property
    def beta(self):
        return torch.sigmoid(self._beta) * 0.9 + 0.1 if self._beta is not None else None

    @property
    def gamma(self):
        return (
            torch.sigmoid(self._gamma) * 0.09 + 0.01
            if self._gamma is not None
            else None
        )

    @property
    def mu(self):
        return torch.sigmoid(self._mu) * 0.09 + 0.01 if self._mu is not None else None

    def init_xavier(self):
        torch.manual_seed(self.retain_seed)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(init_weights)


# Define SIRD model differential equations
def SIRD_model(u, t, beta, gamma, mu, N):
    S, I, R, D = u
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - (gamma + mu) * I
    dRdt = gamma * I
    dDdt = mu * I
    return [dSdt, dIdt, dRdt, dDdt]


# Prepare PyTorch tensors from the data
def prepare_tensors(data, device):
    t = (
        tensor(range(1, len(data) + 1), dtype=torch.float32)
        .view(-1, 1)
        .to(device)
        .requires_grad_(True)
    )
    S = tensor(data["S(t)"].values, dtype=torch.float32).view(-1, 1).to(device)
    I = tensor(data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
    R = tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = (
        tensor(data["cumulative_deceased"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    return t, S, I, R, D


# Split and scale the data into training and validation sets
def split_and_scale_data(data, train_size, features, device):
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

    t_train, S_train, I_train, R_train, D_train = prepare_tensors(
        scaled_train_data, device
    )
    t_val, S_val, I_val, R_val, D_val = prepare_tensors(scaled_val_data, device)

    tensor_data = {
        "train": (t_train, S_train, I_train, R_train, D_train),
        "val": (t_val, S_val, I_val, R_val, D_val),
    }

    return tensor_data, scaler


# Example features and data split
features = ["S(t)", "active_cases", "recovered", "cumulative_deceased"]
train_size = 60  # days

# Assuming 'data' is a DataFrame containing the relevant columns
tensor_data, scaler = split_and_scale_data(data, train_size, features, device)


# PINN loss function
def pinn_loss(tensor_data, model, t, N, device):
    t_train, S_train, I_train, R_train, D_train = tensor_data["train"]
    t_val, S_val, I_val, R_val, D_val = tensor_data["val"]

    S = torch.cat([S_train, S_val], dim=0)
    I = torch.cat([I_train, I_val], dim=0)
    R = torch.cat([R_train, R_val], dim=0)
    D = torch.cat([D_train, D_val], dim=0)

    model_output = model(t)
    S_pred, I_pred, R_pred, D_pred = (
        model_output[:, 0],
        model_output[:, 1],
        model_output[:, 2],
        model_output[:, 3],
    )

    beta, gamma, mu = model.beta, model.gamma, model.mu

    s_t = grad(
        outputs=S_pred,
        inputs=t,
        grad_outputs=torch.ones_like(S_pred),
        create_graph=True,
    )[0]
    i_t = grad(
        outputs=I_pred,
        inputs=t,
        grad_outputs=torch.ones_like(I_pred),
        create_graph=True,
    )[0]
    r_t = grad(
        outputs=R_pred,
        inputs=t,
        grad_outputs=torch.ones_like(R_pred),
        create_graph=True,
    )[0]
    d_t = grad(
        outputs=D_pred,
        inputs=t,
        grad_outputs=torch.ones_like(D_pred),
        create_graph=True,
    )[0]

    dSdt_pred = -beta * S_pred * I_pred / N
    dIdt_pred = beta * S_pred * I_pred / N - (gamma + mu) * I_pred
    dRdt_pred = gamma * I_pred
    dDdt_pred = mu * I_pred

    data_loss_total = torch.mean(
        (S_train - S_pred[: len(S_train)]) ** 2
        + (I_train - I_pred[: len(I_train)]) ** 2
        + (R_train - R_pred[: len(R_train)]) ** 2
        + (D_train - D_pred[: len(D_train)]) ** 2
    )
    physics_loss = torch.mean(
        (s_t - dSdt_pred) ** 2
        + (i_t - dIdt_pred) ** 2
        + (r_t - dRdt_pred) ** 2
        + (d_t - dDdt_pred) ** 2
    )
    initial_condition_loss = torch.mean(
        (S[0] - S_pred[0]) ** 2
        + (I[0] - I_pred[0]) ** 2
        + (R[0] - R_pred[0]) ** 2
        + (D[0] - D_pred[0]) ** 2
    )
    boundary_condition_loss = torch.mean(
        (S[-1] - S_pred[-1]) ** 2
        + (I[-1] - I_pred[-1]) ** 2
        + (R[-1] - R_pred[-1]) ** 2
        + (D[-1] - D_pred[-1]) ** 2
    )
    reg_loss = torch.mean(beta**2 + gamma**2 + mu**2)

    loss = (
        data_loss_total
        + physics_loss
        + initial_condition_loss
        + boundary_condition_loss
        + reg_loss
    )

    return loss


class EarlyStopping:
    """Early stopping utility to stop training when validation loss doesn't improve."""

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


def train_model(
    model, tensor_data, N, device, lr=1e-3, num_epochs=1000, early_stopping_patience=10
):
    loss_history = []

    model_optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    model_scheduler = StepLR(model_optimizer, step_size=5000, gamma=0.9)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        t_train, S_train, I_train, R_train, D_train = tensor_data["train"]

        model_optimizer.zero_grad()
        model_output = model(t_train)
        loss = pinn_loss(tensor_data, model, t_train, N, device)
        loss.backward()
        model_optimizer.step()
        running_loss += loss.item()
        loss_history.append(running_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}")

        model_scheduler.step()

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f"Training completed after {epoch + 1} epochs")
    return model, loss_history


# Instantiate the model
model = SIRDNet(
    inverse=True,
    init_beta=0.5,
    init_gamma=0.01,
    init_mu=0.01,
    retain_seed=42,
    num_layers=6,
    hidden_neurons=32,
).to(device)

# Train the model
N = data["population"].values[0]
trained_model, loss_history = train_model(
    model,
    tensor_data,
    N,
    device,
    lr=1e-3,
    num_epochs=100000,
    early_stopping_patience=50,
)

# Plot the log loss history
plt.plot(np.log(np.array(loss_history)))
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Log Loss History")
plt.tight_layout()
plt.show()

# Extract the model parameters
beta = trained_model.beta.item()
gamma = trained_model.gamma.item()
mu = trained_model.mu.item()

# Print the model parameters
print(f"Estimated beta: {beta:.3f}")
print(f"Estimated gamma: {gamma:.3f}")
print(f"Estimated mu: {mu:.3f}")

# plot the infected cases prediction vs actual data on the training set
model.eval()
with torch.no_grad():
    predictions = model(tensor_data["train"][0]).cpu()

# extract the predictions and actual data
S_pred, I_pred, R_pred, D_pred = (
    predictions[:, 0],
    predictions[:, 1],
    predictions[:, 2],
    predictions[:, 3],
)  # SIRD predictions
S_true, I_true, R_true, D_true = (
    tensor_data["train"][1].cpu(),
    tensor_data["train"][2].cpu(),
    tensor_data["train"][3].cpu(),
    tensor_data["train"][4].cpu(),
)  # SIRD actual data

# rescale the data
rescaled_pred = scaler.inverse_transform(
    np.concatenate(
        [
            S_pred.numpy().reshape(-1, 1),
            I_pred.numpy().reshape(-1, 1),
            R_pred.numpy().reshape(-1, 1),
            D_pred.numpy().reshape(-1, 1),
        ],
        axis=1,
    )
)

# rescale the data
rescaled_true = scaler.inverse_transform(
    np.concatenate(
        [
            S_true.numpy().reshape(-1, 1),
            I_true.numpy().reshape(-1, 1),
            R_true.numpy().reshape(-1, 1),
            D_true.numpy().reshape(-1, 1),
        ],
        axis=1,
    )
)

# extract the infected cases predictions and actual data
I_pred, I_true = rescaled_pred[:, 1], rescaled_true[:, 1]

# plot the infected cases predictions vs actual data
plt.plot(I_true, label="Actual")
plt.plot(I_pred, label="Predicted")
plt.xlabel("Days")
plt.ylabel("Active Cases")
plt.title("Active Cases Prediction vs Actual")
plt.legend()
plt.tight_layout()
plt.show()
