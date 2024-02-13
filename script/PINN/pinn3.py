import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set the default style
mpl.style.use("fivethirtyeight")
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams["font.family"] = "serif"
# mpl.rcParams["text.usetex"] = True
# mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["figure.figsize"] = [15, 8]
mpl.rcParams["figure.autolayout"] = True
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "0.75"
mpl.rcParams["font.size"] = 14
mpl.rcParams["legend.fontsize"] = "medium"


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


df = load_and_preprocess_data("../../data/region_daily_data/East of England.csv")

start_date = "2020-04-01"
end_date = "2021-01-31"
mask = (df["date"] >= start_date) & (df["date"] <= end_date)
training_data = df.loc[mask]

transformer = MinMaxScaler()

# Select the columns to scale
columns_to_scale = [
    "cumulative_confirmed",
    "cumulative_deceased",
    "recovered",
    "active_cases",
    "S(t)",
]

# Fit the scaler to the training data
transformer.fit(training_data[columns_to_scale])

# Transform the training data
training_data[columns_to_scale] = transformer.transform(training_data[columns_to_scale])


# Convert columns to tensors
S_data = (
    torch.tensor(training_data["S(t)"].values, dtype=torch.float32)
    .view(-1, 1)
    .to(device)
)
t_data = (
    torch.tensor(range(len(training_data)), dtype=torch.float32)
    .view(-1, 1)
    .requires_grad_(True)
    .to(device)
)
I_data = (
    torch.tensor(training_data["active_cases"].values, dtype=torch.float32)
    .view(-1, 1)
    .to(device)
)
R_data = (
    torch.tensor(training_data["recovered"].values, dtype=torch.float32)
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
                m.bias.data.fill_(0)

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
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=10000, gamma=0.1
    # )  # Adjust as needed
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=500, factor=0.5, min_lr=1e-2, verbose=True
    )
    early_stopping = EarlyStopping(patience=200)
    history = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(t_data)
        loss = sir_loss(predictions, SIR_tensor, t_data, df["population"].iloc[0])
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
n_hidden_layers = 6
neurons = 50
regularization_param = 0.0001  # Example regularization parameter
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

# plt.plot(time_points, S_actual, 'b', label='Susceptible Actual', linewidth=2)
# plt.plot(time_points, S_pred, 'b--', label='Susceptible Predicted', linewidth=2)

plt.plot(time_points, I_actual, "r", label="Infected Actual", linewidth=2)
plt.plot(time_points, I_pred, "r--", label="Infected Predicted", linewidth=2)
plt.xlabel("Days since: 2020-04-01")
plt.ylabel("Population")
plt.title("SIR Model Predictions vs. Actual Data")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("../../images/sir_model_predictions2.pdf")


plt.plot(time_points, R_actual, "g", label="Recovered Actual", linewidth=2)
plt.plot(time_points, R_pred, "g--", label="Recovered Predicted", linewidth=2)
plt.xlabel("Days since: 2020-04-01")
plt.ylabel("Population")
plt.title("SIR Model Predictions vs. Actual Data")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("../../images/sir_model_predictions2.pdf")

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

print(f"Infected - MAE: {I_mae:.4f}, MSE: {I_mse:.4f}, RMSE: {I_rmse:.4f}, MAPE: {I_mape:.2f}%")
print(f"Deceased - MAE: {D_mae:.4f}, MSE: {D_mse:.4f}, RMSE: {D_rmse:.4f}, MAPE: {D_mape:.2f}%")


# Save the model
torch.save(model, "../../models/sir_model2.pth")
