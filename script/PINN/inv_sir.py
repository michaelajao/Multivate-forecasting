import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler


plt.style.use("seaborn-v0_8-white")  # Use seaborn style for plots

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_preprocess_data(filepath):
    try:
        # Load data from a CSV file
        df = pd.read_csv(filepath)

        # Ensure the 'date', 'cumulative_confirmed', and 'cumulative_deceased' columns exist
        required_columns = [
            "date",
            "cumulative_confirmed",
            "cumulative_deceased",
            "population",
        ]
        if not all(column in df.columns for column in required_columns):
            raise ValueError("Missing required columns in the dataset")

        # Convert 'date' column to datetime format
        df["date"] = pd.to_datetime(df["date"])

        # Calculate the number of days since the start of the dataset
        df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

        # Smooth the 'cumulative_confirmed' and 'cumulative_deceased' with a 7-day rolling average
        for col in ["cumulative_confirmed", "cumulative_deceased"]:
            df[col] = (
                df[col].rolling(window=7, min_periods=1).mean().fillna(0).astype(int)
            )

        # Calculate recovered cases assuming a fixed recovery period
        recovery_period = 21
        df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df[
            "cumulative_deceased"
        ].shift(recovery_period)

        # Calculate the number of active cases
        df["active_cases"] = (
            df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]
        )

        # Calculate the susceptible population (S(t))
        df["S(t)"] = (
            df["population"]
            - df["active_cases"]
            - df["recovered"]
            - df["cumulative_deceased"]
        )

        # Fill any remaining missing values with 0
        df.fillna(0, inplace=True)

        df = df[df["date"] >= "2020-04-01"]

        return df
    except FileNotFoundError:
        print("File not found. Please check the filepath and try again.")
    except pd.errors.EmptyDataError:
        print("No data found. Please check the file content.")
    except ValueError as e:
        print(e)


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


df = load_and_preprocess_data("../../data/region_daily_data/East Midlands.csv")


# Use the function to split the data
train_df, val_df, test_df = split_time_series_data(
    df, train_size=0.7, val_size=0.15, test_size=0.15
)

# Optionally, select a specific size for training (e.g., first 30 data points from the training set)
train_df_selected = train_df.copy()


def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())


# Apply normalization
# train_df_selected["cumulative_confirmed_normalized"] = min_max_normalize(
#     train_df_selected["cumulative_confirmed"]
# )
# train_df_selected["cumulative_deceased_normalized"] = min_max_normalize(
#     train_df_selected["cumulative_deceased"]
# )

transform = MinMaxScaler()
train_df_selected[
    ["cumulative_confirmed_normalized", "cumulative_deceased_normalized"]
] = transform.fit_transform(train_df_selected[["cumulative_confirmed", "cumulative_deceased"]])

t_data_tensor = (
    torch.tensor(range(len(train_df_selected)), dtype=torch.float32)
    .view(-1, 1)
    .to(device)
)
I_data_tensor = (
    torch.tensor(
        train_df_selected["cumulative_confirmed_normalized"].values, dtype=torch.float32
    )
    .view(-1, 1)
    .to(device)
)
R_data_tensor = (
    torch.tensor(
        train_df_selected["cumulative_deceased_normalized"].values, dtype=torch.float32
    )
    .view(-1, 1)
    .to(device)
)
t_data_tensor.requires_grad = True


class PINN(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons):
        super(PINN, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = n_hidden_layers
        self.neurons = neurons

        # Define the network layers
        layers = [nn.Linear(input_dimension, neurons)]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons)]
        layers += [nn.Linear(neurons, output_dimension)]

        self.layers = nn.ModuleList(layers)

        # Apply Xavier initialization to all linear layers
        self.init_xavier()

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.tanh(layer(x))  # Use F.tanh for applying Tanh activation explicitly
        x = self.layers[-1](x)  # No activation on the output layer
        # Assuming the last two outputs are beta(t) and gamma for simplicity in this example
        S, I, R, beta_t, gamma = (
            x[:, 0],
            x[:, 1],
            x[:, 2],
            torch.sigmoid(x[:, 3]),
            torch.sigmoid(x[:, 4]),
        )
        return S, I, R, beta_t, gamma

    def init_xavier(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)


def loss_function(predictions, t_data, I_data, R_data, N, gamma):
    S_pred, I_pred, R_pred, beta_t_pred, _ = predictions
    # Data-driven loss (Mean Squared Error)
    loss_data = torch.mean((I_pred - I_data) ** 2) + torch.mean((R_pred - R_data) ** 2)

    # Compute derivatives of S, I, R with respect to time
    S_t = torch.autograd.grad(
        S_pred, t_data, grad_outputs=torch.ones_like(S_pred), create_graph=True
    )[0]
    I_t = torch.autograd.grad(
        I_pred, t_data, grad_outputs=torch.ones_like(I_pred), create_graph=True
    )[0]
    R_t = torch.autograd.grad(
        R_pred, t_data, grad_outputs=torch.ones_like(R_pred), create_graph=True
    )[0]
    # SIR model equations
    dSdt = -beta_t_pred * S_pred * I_pred / N
    dIdt = beta_t_pred * S_pred * I_pred / N - gamma * I_pred
    dRdt = gamma * I_pred

    # Physics-informed loss
    loss_sir = (
        torch.mean((S_t - dSdt) ** 2)
        + torch.mean((I_t - dIdt) ** 2)
        + torch.mean((R_t - dRdt) ** 2)
    )

    # Total loss
    total_loss = loss_data + loss_sir

    return total_loss

loss_history = []

N = df["population"].iloc[0]  # Total population, assuming constant
gamma = 1 / 10  # Assuming a constant recovery rate, adjust as needed

# Initialize the model
model = PINN(input_dimension=1, output_dimension=5, n_hidden_layers=4, neurons=50).to(
    device
)

# Optimizer setup
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
epochs = 10000

# Modify the training loop to store the loss value at each epoch
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(t_data_tensor)
    loss = loss_function(predictions, t_data_tensor, I_data_tensor, R_data_tensor, N, gamma)
    loss.backward()
    optimizer.step()
    
    # Store the loss
    loss_history.append(loss.item())
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Plotting the training loss
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
# Remember to switch the model to evaluation mode once training is complete


# Convert tensors to numpy for plotting
t_days = t_data_tensor.cpu().detach().numpy()

actual_I = I_data_tensor.detach().cpu().numpy().flatten()
actual_R = R_data_tensor.detach().cpu().numpy().flatten()

# Get model predictions for plotting
model.eval()
with torch.no_grad():
    predictions = model(t_data_tensor)
    S_pred, I_pred, R_pred, beta_t_pred, gamma_pred = predictions


# Plotting the actual vs predicted data
plt.figure(figsize=(15, 6))

# Infected
plt.subplot(1, 2, 1)
plt.plot(t_days, actual_I, label='Actual Infected')
plt.plot(t_days, I_pred.cpu().numpy(), label='Predicted Infected', linestyle='--')
plt.xlabel('Days since start')
plt.ylabel('Infected Population')
plt.title('Actual vs Predicted Infected')
plt.legend()

# Recovered
plt.subplot(1, 2, 2)
plt.plot(t_days, actual_R, label='Actual Recovered')
plt.plot(t_days, R_pred.cpu().numpy(), label='Predicted Recovered', linestyle='--')
plt.xlabel('Days since start')
plt.ylabel('Recovered Population')
plt.title('Actual vs Predicted Recovered')
plt.legend()

plt.tight_layout()
plt.show()


beta_t_pred_rescaled = beta_t_pred.cpu().numpy()

plt.figure(figsize=(10, 5))
plt.plot(t_days, beta_t_pred_rescaled, label=r'Estimated $\beta(t)$', color='orange')
plt.xlabel('Days since start')
plt.ylabel(r'Transmission Rate $\beta(t)$')
plt.title('Estimated Time-Varying Transmission Rate')
plt.legend()
plt.show()



def rescale(normalized_data, original_data):
    """Rescale the normalized data back to the original scale."""
    min_val = original_data.min()
    max_val = original_data.max()
    return normalized_data * (max_val - min_val) + min_val


def visualize_cumulative_cases(
    t_data, actual_infections, actual_deaths, model_predictions, N
):
    """
    Visualizes the actual vs. predicted cumulative cases and deaths.

    Args:
        t_data (Tensor): Days since the start of the dataset.
        actual_infections (Tensor): Actual cumulative infections.
        actual_deaths (Tensor): Actual cumulative deaths.
        model_predictions (Tensor): Model's predictions [S, I, R] for each day.
        N (int): Total population.
    """
    # Detach tensors from the computation graph and move to CPU
    t_data_np = t_data.detach().cpu().numpy().reshape(-1)
    actual_infections_np = actual_infections.detach().cpu().numpy().reshape(-1)
    actual_deaths_np = actual_deaths.detach().cpu().numpy().reshape(-1)

    # Rescale predictions if they were normalized
    I_pred_rescaled = (
        rescale(model_predictions[:, 1], actual_infections)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    R_pred_rescaled = (
        rescale(model_predictions[:, 2], actual_deaths)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    # Plotting
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(
        t_data_np,
        actual_infections_np,
        label="Actual Infections",
        marker="o",
        linestyle="-",
        color="blue",
    )
    plt.plot(
        t_data_np,
        I_pred_rescaled,
        label="Predicted Infections",
        marker="x",
        linestyle="--",
        color="red",
    )
    plt.xlabel("Days")
    plt.ylabel("Cumulative Infections")
    plt.title("Cumulative Infections: Actual vs Predicted")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        t_data_np,
        actual_deaths_np,
        label="Actual Deaths",
        marker="o",
        linestyle="-",
        color="green",
    )
    plt.plot(
        t_data_np,
        R_pred_rescaled,
        label="Predicted Deaths",
        marker="x",
        linestyle="--",
        color="purple",
    )
    plt.xlabel("Days")
    plt.ylabel("Cumulative Deaths")
    plt.title("Cumulative Deaths: Actual vs Predicted")
    plt.legend()

    plt.tight_layout()
    plt.show()


visualize_cumulative_cases(
    t_data_tensor, I_data_tensor, R_data_tensor, final_predictions, N
)

# visualize_cumulative_cases(
#     t_data_tensor, I_data_tensor, R_data_tensor, final_predictions, N
# )


def visualize_parameters(t_data, beta_estimates, gamma_estimates):
    """
    Visualizes the estimated parameters over time.

    Args:
        t_data (Tensor): Days since the start of the dataset.
        beta_estimates (Tensor): Estimated transmission rates.
        gamma_estimates (Tensor): Estimated recovery rates.
    """
    t_data_np = t_data.cpu().numpy().reshape(-1)
    beta_estimates_np = beta_estimates.cpu().detach().numpy().reshape(-1)
    gamma_estimates_np = gamma_estimates.cpu().detach().numpy().reshape(-1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(
        t_data_np,
        beta_estimates_np,
        label=r"Estimated $\beta(t)$",
        marker="o",
        linestyle="-",
        color="orange",
    )
    plt.xlabel("Days")
    plt.ylabel(r"Transmission Rate ($\beta$)")
    plt.title("Estimated Transmission Rate Over Time")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        t_data_np,
        gamma_estimates_np,
        label=r"Estimated $\gamma(t)$",
        marker="o",
        linestyle="-",
        color="cyan",
    )
    plt.xlabel("Days")
    plt.ylabel(r"Recovery Rate ($\gamma$)")
    plt.title("Estimated Recovery Rate Over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_training_loss(history):
    """
    Visualizes the training loss over epochs.

    Args:
        history (list): Training loss recorded at each epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, label="Training Loss", color="purple", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


# Visualize Cumulative Cases and Deaths
visualize_cumulative_cases(
    t_data_tensor, I_data_tensor, R_data_tensor, final_predictions, N
)
visualize_training_loss(history)

# # Visualize Parameters (Beta and Gamma)
# visualize_parameters(t_data_tensor, final_beta, final_gamma)

# Visualize Training Loss
visualize_training_loss(history)


def calculate_mae(actual, predicted):
    return torch.mean(torch.abs(actual - predicted))


def calculate_mse(actual, predicted):
    return torch.mean((actual - predicted) ** 2)


def calculate_rmse(actual, predicted):
    return torch.sqrt(calculate_mse(actual, predicted))


def calculate_mape(actual, predicted):
    return torch.mean(torch.abs((actual - predicted) / actual)) * 100


future_days = [3, 5, 7, 14]
future_predictions = {}

for days in future_days:
    future_time_points = (
        torch.arange(t_data_tensor.max() + 1, t_data_tensor.max() + 1 + days, 1)
        .view(-1, 1)
        .to(device)
    )
    with torch.no_grad():
        future_sir, future_beta, future_gamma = my_network(future_time_points)
    future_predictions[days] = (future_sir, future_beta, future_gamma)

# Example for evaluating 3-day ahead predictions
# You would replace future_actual_infections and future_actual_deaths with your actual future data
future_actual_infections = (
    torch.tensor([...], dtype=torch.float32).view(-1, 1).to(device)
)
future_actual_deaths = torch.tensor([...], dtype=torch.float32).view(-1, 1).to(device)

future_sir, _, _ = future_predictions[3]  # For 3-day predictions
I_pred, R_pred = future_sir[:, 1], future_sir[:, 2]  # Extracting I and R predictions

# Calculate metrics for infections
mae_infections = calculate_mae(future_actual_infections, I_pred)
mse_infections = calculate_mse(future_actual_infections, I_pred)
rmse_infections = calculate_rmse(future_actual_infections, I_pred)
mape_infections = calculate_mape(future_actual_infections, I_pred)

# Calculate metrics for deaths
mae_deaths = calculate_mae(future_actual_deaths, R_pred)
mse_deaths = calculate_mse(future_actual_deaths, R_pred)
rmse_deaths = calculate_rmse(future_actual_deaths, R_pred)
mape_deaths = calculate_mape(future_actual_deaths, R_pred)

print(
    f"Metrics for 3-day ahead predictions - Infections: MAE={mae_infections}, MSE={mse_infections}, RMSE={rmse_infections}, MAPE={mape_infections}%"
)
print(
    f"Metrics for 3-day ahead predictions - Deaths: MAE={mae_deaths}, MSE={mse_deaths}, RMSE={rmse_deaths}, MAPE={mape_deaths}%"
)
