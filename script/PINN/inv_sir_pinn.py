import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
train_df_selected = train_df.head(30)


def df_to_tensors(df):
    """
    Converts specified columns of a DataFrame into PyTorch tensors.

    Args:
        df (pd.DataFrame): Input DataFrame with time series data.

    Returns:
        tuple: Tensors for days_since_start, cumulative_confirmed, and cumulative_deceased.
    """

    t_data_tensor = (
        torch.tensor(range(len(df)), dtype=torch.float32)
        .view(-1, 1)
        .requires_grad_(True)
        .to(device)
    )
    I_data_tensor = (
        torch.tensor(df["cumulative_confirmed"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    R_data_tensor = (
        torch.tensor(df["cumulative_deceased"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )

    return t_data_tensor, I_data_tensor, R_data_tensor


# Convert the selected training DataFrame to tensors
t_data_tensor, I_data_tensor, R_data_tensor = df_to_tensors(train_df_selected)


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
        self.beta = nn.Linear(self.neurons, 1)  # For learning transmission rate
        self.gamma = nn.Linear(self.neurons, 1)  # For learning recovery rate

        self.init_xavier()

    def forward(self, x, S0):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        sir = self.output_layer(x)
        beta = torch.sigmoid(self.beta(x))  # Ensuring beta is in (0, 1)
        gamma = torch.sigmoid(self.gamma(x))  # Ensuring gamma is in (0, 1)
        R0 = self.beta / self.gamma
        Rt = R0 * S0 / self.N
        return sir, beta, gamma, R0, Rt

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


def compute_loss(
    model,
    t,
    cumulative_infections,
    cumulative_deaths,
    S0,
    I0,
    R0,
    N,
    weight_physics=1.0,
    weight_data=1.0,
    weight_initial=1.0,
):
    sir, beta, gamma, R0_pred, Rt_pred = model(t, S0)
    S, I, R = sir[:, 0], sir[:, 1], sir[:, 2]

    S_t = torch.autograd.grad(S.sum(), t, create_graph=True)[0]
    I_t = torch.autograd.grad(I.sum(), t, create_graph=True)[0]
    R_t = torch.autograd.grad(R.sum(), t, create_graph=True)[0]

    dSdt = -beta.squeeze() * S * I / N
    dIdt = beta.squeeze() * S * I / N - gamma.squeeze() * I
    dRdt = gamma.squeeze() * I

    loss_physics = (
        torch.mean((S_t - dSdt) ** 2)
        + torch.mean((I_t - dIdt) ** 2)
        + torch.mean((R_t - dRdt) ** 2)
    )
    loss_data = torch.mean((I - cumulative_infections) ** 2) + torch.mean(
        (R - cumulative_deaths) ** 2
    )
    
        # Initial conditions loss
    initial_conditions_loss = weight_initial * (torch.square(S[0] - S0) + torch.square(I[0] - I0) + torch.square(R[0] - R0))

    # Weighted total loss
    total_loss = (
        weight_physics * loss_physics
        + weight_data * loss_data
        + initial_conditions_loss
        + model.regularization()
    )

    return total_loss

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
            

# Initialize the PINN model
N = train_df['population'].iloc[0]

my_network = NeuralNet(
    input_dimension=1,
    output_dimension=3,  # S, I, R compartments
    n_hidden_layers=2,
    neurons=65,
    regularization_param=0.0001,
    regularization_exp=2,
    retrain_seed=42,
).to(device)

def train_PINN(model, t_data, cumulative_infections_tensor, cumulative_deaths_tensor, num_epochs=5000, lr=0.01, N=1000000):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.1)
    early_stopping = EarlyStopping(patience=500, verbose=True)
    history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions, beta, gamma = model(t_data)
        loss = compute_loss(model, t_data, cumulative_infections_tensor, cumulative_deaths_tensor, N)
        reg_loss = model.regularization()  # Calculate regularization loss if applicable
        total_loss = loss + reg_loss  # Combine primary and regularization losses
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        history.append(total_loss.item())
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}, Learning rate: {scheduler.get_last_lr()[0]}")

        early_stopping(total_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    return model, history, predictions, beta, gamma

