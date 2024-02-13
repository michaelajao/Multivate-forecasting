import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-white")
plt.rcParams["font.size"] = 12
plt.rcParams["figure.figsize"] = (12, 6)

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # Check required columns
    required_columns = [
        "date",
        "cumulative_confirmed",
        "cumulative_deceased",
        "population",
    ]
    if not all(column in df.columns for column in required_columns):
        raise ValueError("Missing required columns in the dataset")

    # Convert date to datetime and create a day counter
    df["date"] = pd.to_datetime(df["date"])
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

    # Calculate rolling averages and discard the initial rows with insufficient data for rolling calculation
    for col in ["cumulative_confirmed", "cumulative_deceased"]:
        df[col] = df[col].rolling(window=7, min_periods=7).mean()

    # Calculate other columns after dropping initial NaN values
    df = df.dropna().reset_index(drop=True)
    recovery_period = 21
    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df[
        "cumulative_deceased"
    ].shift(recovery_period)
    df["active_cases"] = (
        df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]
    )
    df["S(t)"] = (
        df["population"]
        - df["active_cases"]
        - df["recovered"]
        - df["cumulative_deceased"]
    )

    # Normalize the required fields
    scaler = StandardScaler()
    df[
        [
            "cumulative_confirmed",
            "cumulative_deceased",
            "recovered",
            "active_cases",
            "S(t)",
        ]
    ] = scaler.fit_transform(
        df[
            [
                "cumulative_confirmed",
                "cumulative_deceased",
                "recovered",
                "active_cases",
                "S(t)",
            ]
        ]
    )

    return df, scaler


df, scaler = load_and_preprocess_data("../../data/region_daily_data/East Midlands.csv")
df = df[df["date"] >= "2020-04-01"].reset_index(drop=True)


def split_time_series_data(df, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Splits the DataFrame into training, validation, and test sets while maintaining the time series order.

    Args:
        df (pd.DataFrame): The input DataFrame with time series data.
        train_size (float): Proportion of the dataset to allocate to training.
        val_size (float): Proportion of the dataset to allocate to validation.
        test_size (float): Proportion of the dataset to allocate to testing.

    Returns:
        tuple: Three DataFrames corresponding to the training, validation, and test sets.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size, val_size, and test_size should sum to 1.")

    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]

    return train_data, val_data, test_data


train_df, val_df, test_df = split_time_series_data(
    df, train_size=0.7, val_size=0.15, test_size=0.15
)


def create_dataset(df):
    # Time tensor based on the length of the dataframe
    t_tensor = (
        torch.tensor(range(len(df)), dtype=torch.float32, requires_grad=True)
        .view(-1, 1)
        .to(device)
    )

    # Confirmed cases tensor
    confirmed_tensor = (
        torch.tensor(df["cumulative_confirmed"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )

    # Susceptible population tensor
    susceptible_tensor = (
        torch.tensor(df["S(t)"].values, dtype=torch.float32).view(-1, 1).to(device)
    )

    # Recovered cases tensor
    recovered_tensor = (
        torch.tensor(df["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    )

    # Combine susceptible, infected (confirmed), and recovered tensors into a single SIR tensor
    SIR_tensor = torch.cat(
        [susceptible_tensor, confirmed_tensor, recovered_tensor], dim=1
    )

    return t_tensor, SIR_tensor


# Create datasets for training, validation, and testing
t_train, SIR_train = create_dataset(train_df)
t_val, SIR_val = create_dataset(val_df)
t_test, SIR_test = create_dataset(test_df)
# train_df = train_df.head(30).copy()
# Check the shape of the tensors
print(f"Training set shapes - t_train: {t_train.shape}, SIR_train: {SIR_train.shape}")
print(f"Validation set shapes - t_val: {t_val.shape}, SIR_val: {SIR_val.shape}")
print(f"Testing set shapes - t_test: {t_test.shape}, SIR_test: {SIR_test.shape}")


class NeuralNet(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        n_hidden_layers,
        neurons,
        regularization_param,
        regularization_exp,
        retrain_seed,
    ):
        super(NeuralNet, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers
        self.activation = nn.Tanh()
        self.regularization_param = regularization_param
        self.regularization_exp = regularization_exp
        self.retrain_seed = retrain_seed

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)]
        )
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.init_xavier()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)

    # Initialize the neural network with Xavier Initialization
    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if "weight" in name:
                reg_loss += torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss


def sir_loss(model_output, SIR_tensor, t, N, beta=0.25, gamma=0.15):
    S_pred, I_pred, R_pred = model_output[:, 0], model_output[:, 1], model_output[:, 2]

    S_t = torch.autograd.grad(
        S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True
    )[0]
    I_t = torch.autograd.grad(
        I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True
    )[0]
    R_t = torch.autograd.grad(
        R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True
    )[0]

    dSdt = -beta * S_pred * I_pred / N
    dIdt = (beta * S_pred * I_pred) / N - (gamma * I_pred)
    dRdt = gamma * I_pred

    loss = (
        torch.mean((S_t - dSdt) ** 2)
        + torch.mean((I_t - dIdt) ** 2)
        + torch.mean((R_t - dRdt) ** 2)
    )
    loss += torch.mean((model_output - SIR_tensor) ** 2)

    return loss


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})."
            )
        self.val_loss_min = val_loss


def train_PINN(model, t_data, SIR_tensor, num_epochs=5000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)  # Adjust gamma as needed
    early_stopping = EarlyStopping(patience=10)
    history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(t_data)
        loss = sir_loss(predictions, SIR_tensor, t_data, params["N"])
        reg_loss = model.regularization()  # Un-comment and compute regularization loss
        total_loss = loss + reg_loss  # Combine primary loss and regularization loss
        total_loss.backward()
        optimizer.step()
        # scheduler.step()
        # , LR: {scheduler.get_last_lr()[0]}'
        history.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}")

        if early_stopping(total_loss):
            print("Early stopping")
            break

    return model, history


# Model parameters
params = {
    "input_dimension": 1,
    "output_dimension": 3,
    "n_hidden_layers": 3,
    "neurons": 10,
    "regularization_param": 0.001,
    "regularization_exp": 2,
    "retrain_seed": 42,
    "N": df["population"].values[0],
}

# Initialize the model
model = NeuralNet(
    params["input_dimension"],
    params["output_dimension"],
    params["n_hidden_layers"],
    params["neurons"],
    params["regularization_param"],
    params["regularization_exp"],
    params["retrain_seed"],
).to(device)


# Train the model
model, history = train_PINN(model, t_train, SIR_train, num_epochs=10000, lr=0.01)


plt.plot(history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()


def smooth_curve(points, factor=0.8):
    """Smooths a list of points using exponential moving average."""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# Apply smoothing to the training history
smoothed_history = smooth_curve(history)


plt.plot(smoothed_history, label="Training Loss (Smoothed)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.yscale("log")  # Use logarithmic scale to better visualize changes
plt.legend()
plt.grid(True)
plt.show()


def evaluate_model(model, t_data, SIR_data, dataset_name="Validation"):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        predictions = model(t_data)
        # Optionally calculate a simpler loss metric that doesn't involve gradients
        # For example, mean squared error (MSE) on the predictions vs actual SIR data
        mse_loss = torch.mean((predictions - SIR_data) ** 2).item()
    print(f"{dataset_name} MSE Loss: {mse_loss:.4f}")
    return predictions


# Evaluate on validation set
val_predictions = evaluate_model(model, t_val, SIR_val, "Validation")

# Evaluate on test set
test_predictions = evaluate_model(model, t_test, SIR_test, "Test")


def plot_predictions(
    t_data, actual_data, predictions, title="Predictions vs Actual Data"
):
    plt.figure(figsize=(12, 6))
    # Detach tensors and convert to numpy for plotting
    t_data_np = t_data.detach().cpu().numpy()
    actual_data_np = actual_data.detach().cpu().numpy()
    predictions_np = predictions.detach().cpu().numpy()

    plt.plot(
        t_data_np, actual_data_np[:, 1], "r", label="Actual Confirmed Cases"
    )  # Actual confirmed cases
    plt.plot(
        t_data_np, predictions_np[:, 1], "b--", label="Predicted Confirmed Cases"
    )  # Predicted confirmed cases
    plt.xlabel("Days")
    plt.ylabel("Normalized Cases")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot for validation set
plot_predictions(
    t_val, SIR_val, val_predictions, "Validation Set Predictions vs Actual Data"
)

# Plot for test set
plot_predictions(
    t_test, SIR_test, test_predictions, "Test Set Predictions vs Actual Data"
)
