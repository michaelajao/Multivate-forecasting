import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set the default style
# plt.style.use("fivethirtyeight")
# plt.style.use("seaborn-v0_8-poster")
# plt.rcParams.update(
#     {
#         "lines.linewidth": 2,
#         "font.family": "serif",
#         "axes.titlesize": 20,
#         "axes.labelsize": 14,
#         "figure.figsize": [15, 8],
#         "figure.autolayout": True,
#         "axes.spines.top": False,
#         "axes.spines.right": False,
#         "axes.grid": True,
#         "grid.color": "0.75",
#         "legend.fontsize": "medium",
#     }
# )

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

# Device configuration
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data = pd.read_csv("../../data/hos_data/merged_data.csv")


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
    df["susceptible"] = (
        df["population"]
        - df["recovered"]
        - df["cumulative_deceased"]
        - df["cumulative_confirmed"]
    )

    cols_to_smooth = [
        "cumulative_confirmed",
        "cumulative_deceased",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "recovered",
        "active_cases",
        "new_deceased",
        "new_confirmed",
        "susceptible",
    ]

    for col in cols_to_smooth:
        df[col] = df[col].clip(lower=0)

    for col in cols_to_smooth:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    return df


# Load and preprocess the data
data = load_and_preprocess_data(
    "../../data/hos_data/merged_data.csv",
    areaname="South West",
    recovery_period=21,
    start_date="2020-04-01",
    end_date="2020-08-31",
)

region_name = "South West"

# def load_and_preprocess_data(filepath, areaname="Yorkshire and the Humber"):
#     try:
#         # Load data from a CSV file
#         df = pd.read_csv(filepath)
#         df = df[::-1].reset_index(drop=True)  # Reverse dataset if needed

#         df = df[df["nhs_region"] == areaname].reset_index(drop=True)
#         # Ensure the 'date', 'cumulative_confirmed', and 'cumulative_deceased' columns exist
#         required_columns = [
#             "date",
#             "cumulative_confirmed",
#             "cumulative_deceased",
#             "population",
#             "new_confirmed",
#             "new_deceased",
#         ]
#         if not all(column in df.columns for column in required_columns):
#             raise ValueError("Missing required columns in the dataset")

#         # Convert 'date' column to datetime format
#         df["date"] = pd.to_datetime(df["date"])

#         # Calculate the number of days since the start of the dataset
#         df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

#         # Calculate recovered cases assuming a fixed recovery period
#         recovery_period = 21
#         df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df[
#             "cumulative_deceased"
#         ].shift(recovery_period)

#         # Calculate the number of active cases
#         df["active_cases"] = (
#             df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]
#         )

#         # Calculate the susceptible population (S(t))
#         df["S(t)"] = (
#             df["population"]
#             - df["recovered"]
#             - df["active_cases"]
#             - df["cumulative_deceased"]
#         )

#         # Smooth the 'cumulative_confirmed' and 'cumulative_deceased' with a 7-day rolling average
#         for col in [
#             "new_confirmed",
#             "new_deceased",
#             "cumulative_confirmed",
#             "cumulative_deceased",
#             "recovered",
#             "active_cases",
#             "S(t)",
#         ]:
#             df[col] = (
#                 df[col].rolling(window=7, min_periods=1).mean().fillna(0).astype(int)
#             )

#         # Fill any remaining missing values with 0
#         df.fillna(0, inplace=True)

#     except FileNotFoundError:
#         print("File not found. Please check the filepath and try again.")
#     except pd.errors.EmptyDataError:
#         print("No data found. Please check the file content.")
#     except ValueError as e:
#         print(e)

#         return df


# def get_region_name_from_filepath(filepath):

#     base = os.path.basename(filepath)
#     return os.path.splitext(base)[0]


# path = "../../data/region_daily_data/Yorkshire and the Humber.csv"
# region_name = get_region_name_from_filepath(path)
# df = load_and_preprocess_data(f"../../data/region_daily_data/{region_name}.csv")

# start_date = "2020-04-01"
# end_date = "2020-08-31"
# mask = (df["date"] >= start_date) & (df["date"] <= end_date)
# data = data.loc[mask]

transformer = MinMaxScaler()

# Select the columns to scale
columns_to_scale = ["susceptible", "active_cases", "recovered"]

# Fit the scaler to the training data
transformer.fit(data[columns_to_scale])

# Transform the training data
data[columns_to_scale] = transformer.transform(data[columns_to_scale])


# Convert columns to tensors
S_data = (
    torch.tensor(data["susceptible"].values, dtype=torch.float32)
    .view(-1, 1)
    .to(device)
)
t_data = (
    torch.tensor(range(len(data)), dtype=torch.float32)
    .view(-1, 1)
    .requires_grad_(True)
    .to(device)
)
I_data = (
    torch.tensor(data["active_cases"].values, dtype=torch.float32)
    .view(-1, 1)
    .to(device)
)
R_data = (
    torch.tensor(data["recovered"].values, dtype=torch.float32)
    .view(-1, 1)
    .to(device)
)
SIR_tensor = torch.cat([S_data, I_data, R_data], dim=1).to(device)


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
                m.bias.data.fill_(0.01)

        self.apply(init_weights)

    def regularization(self):
        reg_loss = 0
        for name, param in self.named_parameters():
            if "weight" in name:
                reg_loss += torch.norm(param, self.regularization_exp)
        return self.regularization_param * reg_loss


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


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


def train_PINN(model, t_data, SIR_tensor, num_epochs=5000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=10000, gamma=0.1
    # )  # Adjust as needed
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=500, factor=0.5, min_lr=1e-2, verbose=True
    )
    early_stopping = EarlyStopping(patience=100)
    history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(t_data)
        N =data["population"].iloc[0] / data["population"].iloc[0]
        loss = sir_loss(predictions, SIR_tensor, t_data, N)
        # reg_loss = model.regularization()  # Un-comment and compute regularization loss
        # total_loss = loss + reg_loss  # Combine primary loss and regularization loss
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        # scheduler.step()  # Update the learning rate according to the scheduler
        # LR: {scheduler.get_last_lr()[0]}"
        history.append(loss.item())  # Log the total loss, including regularization
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}"
            )  # Print the current learning rate

        early_stopping(loss.item())  # Pass the total loss to early stopping
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model, history


input_dimension = 1
output_dimension = 3
n_hidden_layers = 5
neurons = 50
regularization_param = 0.001  # Example regularization parameter
regularization_exp = 2  # L2 regularization
retrain_seed = 42

my_network = NeuralNet(
    input_dimension,
    output_dimension,
    n_hidden_layers,
    neurons,
    regularization_exp,
    regularization_param,
    retrain_seed,
).to(device)

# Train the model using the physics-informed loss
model, history = train_PINN(my_network, t_data, SIR_tensor, num_epochs=100000, lr=0.001)

# Plot training history
plt.grid(True, which="both", ls=":")
plt.plot(np.arange(1, len(history) + 1), np.log10(history), label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Log10(Loss)")
plt.title("Training History")
plt.legend()
plt.show()

# Switch the model to evaluation mode
model.eval()

# Generate predictions for the same inputs used during training
with torch.no_grad():
    predictions = model(t_data)
# Extract the predicted S, I, R values
S_pred, I_pred, R_pred = (
    predictions[:, 0].cpu().numpy(),
    predictions[:, 1].cpu().numpy(),
    predictions[:, 2].cpu().numpy(),
)

# Extract the actual S, I, R values from the SIR_tensor
S_actual, I_actual, R_actual = (
    SIR_tensor[:, 0].cpu().numpy(),
    SIR_tensor[:, 1].cpu().numpy(),
    SIR_tensor[:, 2].cpu().numpy(),
)

# Extract the time points from t_tensor for plotting
time_points = t_data.cpu().detach().numpy()

# Plotting the actual vs. predicted data

plt.plot(time_points, I_actual, "r", label="Infected Actual", linewidth=2)
plt.plot(time_points, I_pred, "r--", label="Infected Predicted", linewidth=2)
plt.xlabel("Days since: 2020-04-01")
plt.ylabel("Population")
plt.title(f"SIR Model Predictions vs. Actual Data {region_name}")
plt.legend()
# plt.savefig(f"../../images/sir_model_predictions_{region_name}.pdf")
plt.show()


plt.plot(time_points, R_actual, "g", label="Recovered Actual", linewidth=2)
plt.plot(time_points, R_pred, "g--", label="Recovered Predicted", linewidth=2)
plt.xlabel("Days since: 2020-04-01")
plt.ylabel("Population")
plt.title(f"SIR Model Predictions vs. Actual Data {region_name}")
plt.legend()
# plt.savefig(f"../../images/sir_model_predictions_{region_name}.pdf")
plt.show()


# compute MAE, MSE, RMSE, and MAPE for predictions for infected and death cases


def compute_metrics(actual, predicted):
    epsilon = 1e-1  # Small constant to prevent division by zero
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100
    return mae, mse, rmse, mape


# Assuming I_actual, I_pred, D_actual, D_pred are defined
I_mae, I_mse, I_rmse, I_mape = compute_metrics(I_actual, I_pred)
D_mae, D_mse, D_rmse, D_mape = compute_metrics(R_actual, R_pred)

print(
    f"Infected - MAE: {I_mae:.4f}, MSE: {I_mse:.4f}, RMSE: {I_rmse:.4f}, MAPE: {I_mape:.2f}%"
)
print(
    f"Deceased - MAE: {D_mae:.4f}, MSE: {D_mse:.4f}, RMSE: {D_rmse:.4f}, MAPE: {D_mape:.2f}%"
)

model.state_dict()

# Save the model
torch.save(model, "../../models/sir_model.pth")

full_predicted_data = np.zeros((len(S_pred), transformer.n_features_in_))
full_actual_data = np.zeros((len(S_actual), transformer.n_features_in_))

# Fill in the placeholders with the predicted and actual data
# The order of columns in 'columns_to_scale' is ['recovered', 'active_cases', 'S(t)']
full_predicted_data[:, columns_to_scale.index("susceptible")] = S_pred
full_predicted_data[:, columns_to_scale.index("active_cases")] = I_pred
full_predicted_data[:, columns_to_scale.index("recovered")] = R_pred

full_actual_data[:, columns_to_scale.index("susceptible")] = S_actual
full_actual_data[:, columns_to_scale.index("active_cases")] = I_actual
full_actual_data[:, columns_to_scale.index("recovered")] = R_actual

# Apply inverse transformation
inverse_predicted_data = transformer.inverse_transform(full_predicted_data)
inverse_actual_data = transformer.inverse_transform(full_actual_data)

# Separate the inversely transformed S, I, R values for predicted and actual data
S_pred_transformed = inverse_predicted_data[:, columns_to_scale.index("susceptible")]
I_pred_transformed = inverse_predicted_data[:, columns_to_scale.index("active_cases")]
R_pred_transformed = inverse_predicted_data[:, columns_to_scale.index("recovered")]

S_actual_transformed = inverse_actual_data[:, columns_to_scale.index("susceptible")]
I_actual_transformed = inverse_actual_data[:, columns_to_scale.index("active_cases")]
R_actual_transformed = inverse_actual_data[:, columns_to_scale.index("recovered")]


# Plot for Susceptible (S)

plt.plot(
    time_points, S_actual_transformed, "b", label="Susceptible Actual", linewidth=2
)
plt.plot(
    time_points, S_pred_transformed, "b--", label="Susceptible Predicted", linewidth=2
)
plt.xlabel("Days since: 2020-04-01")
plt.ylabel("Population")
plt.title(f"Susceptible: Predictions vs Actual Data {region_name}")
plt.legend()
plt.savefig(f"../../reports/images/PINN/S_predictions_{region_name}.pdf")
plt.show()


# Plot for Infected (I)

plt.plot(time_points, I_actual_transformed, "r", label="Infected Actual", linewidth=2)
plt.plot(
    time_points, I_pred_transformed, "r--", label="Infected Predicted", linewidth=2
)
plt.xlabel("Days since: 2020-04-01")
plt.ylabel("Population")
plt.title(f"Infected: Predictions vs Actual Data {region_name}")
plt.legend()
plt.savefig(f"../../reports/images/PINN/I_predictions_{region_name}.pdf")
plt.show()


# Plot for Recovered (R)

plt.plot(time_points, R_actual_transformed, "g", label="Recovered Actual", linewidth=2)
plt.plot(
    time_points, R_pred_transformed, "g--", label="Recovered Predicted", linewidth=2
)
plt.xlabel("Days since: 2020-04-01")
plt.ylabel("Population")
plt.title(f"Recovered: Predictions vs Actual Data {region_name}")
plt.legend()
plt.savefig(f"../../reports/images/PINN/R_predictions_{region_name}.pdf")
plt.show()


I_mae, I_mse, I_rmse, I_mape = compute_metrics(I_actual_transformed, I_pred_transformed)
D_mae, D_mse, D_rmse, D_mape = compute_metrics(R_actual_transformed, R_pred_transformed)

print(
    f"Infected - MAE: {I_mae:.4f}, MSE: {I_mse:.4f}, RMSE: {I_rmse:.4f}, MAPE: {I_mape:.2f}%"
)
print(
    f"Recovered - MAE: {D_mae:.4f}, MSE: {D_mse:.4f}, RMSE: {D_rmse:.4f}, MAPE: {D_mape:.2f}%"
)

# rather than NeuralNet for training the model, lets make use of an RNN architecture to train the model
