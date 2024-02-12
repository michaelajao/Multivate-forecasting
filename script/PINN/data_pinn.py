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
        required_columns = ['date', 'cumulative_confirmed', 'cumulative_deceased', 'population']
        if not all(column in df.columns for column in required_columns):
            raise ValueError("Missing required columns in the dataset")
        
        # Convert 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate the number of days since the start of the dataset
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        # Smooth the 'cumulative_confirmed' and 'cumulative_deceased' with a 7-day rolling average
        for col in ['cumulative_confirmed', 'cumulative_deceased']:
            df[col] = df[col].rolling(window=7, min_periods=1).mean().fillna(0).astype(int)
        
        # Calculate recovered cases assuming a fixed recovery period
        recovery_period = 21
        df['recovered'] = df['cumulative_confirmed'].shift(recovery_period) - df['cumulative_deceased'].shift(recovery_period)
        
        # Calculate the number of active cases
        df['active_cases'] = df['cumulative_confirmed'] - df['recovered'] - df['cumulative_deceased']
        
        # Calculate the susceptible population (S(t))
        df['S(t)'] = df['population'] - df['active_cases'] - df['recovered'] - df['cumulative_deceased']
        
        # Fill any remaining missing values with 0
        df.fillna(0, inplace=True)
        
        df = df[df['date'] >= '2020-04-01']
        
        return df
    except FileNotFoundError:
        print("File not found. Please check the filepath and try again.")
    except pd.errors.EmptyDataError:
        print("No data found. Please check the file content.")
    except ValueError as e:
        print(e)



df = load_and_preprocess_data("../../data/region_daily_data/East Midlands.csv") 
data = df.head(30)

# Correct your file path
# data = df[df["S(t)"] > 0].head(30)  # Select first 30 data points

# Convert columns to tensors
S_data = torch.tensor(data['S(t)'].values, dtype=torch.float32).view(-1, 1).to(device)
t_data = torch.tensor(range(len(data)), dtype=torch.float32).view(-1, 1). requires_grad_(True).to(device)
I_data = torch.tensor(data['cumulative_confirmed'].values, dtype=torch.float32).view(-1, 1).to(device)
R_data = torch.tensor(data['cumulative_deceased'].values, dtype=torch.float32).view(-1, 1).to(device)
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

    dSdt = - beta * S_pred * I_pred / N
    dIdt = (beta * S_pred * I_pred) / N - (gamma * I_pred)
    dRdt = gamma * I_pred

    loss = (
        torch.mean((S_t - dSdt) ** 2)
        + torch.mean((I_t - dIdt) ** 2)
        + torch.mean((R_t - dRdt) ** 2)
    )
    loss += torch.mean((model_output - SIR_tensor) ** 2)

    return loss


# def sir_loss(
#     model,
#     t,
#     cumulative_infections,
#     cumulative_deaths,
#     N,
#     beta=0.25, gamma=0.15,
#     weight_physics=1.0,
#     weight_data=1.0,
# ):
#     sir, beta, gamma = model(t)
#     S, I, R = sir[:, 0], sir[:, 1], sir[:, 2]

#     # Compute gradients
#     # grad_outputs = torch.ones(S.shape, device=device)
#     S_t = torch.autograd.grad(S, t, grad_outputs=torch.ones_like(S), create_graph=True)[
#         0
#     ]
#     I_t = torch.autograd.grad(I, t, grad_outputs=torch.ones_like(I), create_graph=True)[
#         0
#     ]
#     R_t = torch.autograd.grad(R, t, grad_outputs=torch.ones_like(R), create_graph=True)[
#         0
#     ]

#     # SIR model equations
#     dSdt = -beta * S * I / N
#     dIdt = beta * S * I / N - gamma * I
#     dRdt = gamma* I

#     # Physics-informed loss
#     loss_physics = (
#         torch.mean((S_t - dSdt) ** 2)
#         + torch.mean((I_t - dIdt) ** 2)
#         + torch.mean((R_t - dRdt) ** 2)
#     )

#     # Data fitting loss
#     loss_data = torch.mean((I - cumulative_infections) ** 2) + torch.mean(
#         (R - cumulative_deaths) ** 2
#     )

#     # Weighted total loss
#     total_loss = weight_physics * loss_physics + weight_data * loss_data

#     return total_loss


def train_PINN(model, t_data, SIR_tensor, num_epochs=5000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=50000, gamma=0.1
    )  # Adjust as needed
    early_stopping = EarlyStopping(patience=1000, verbose=True)
    history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(t_data)
        loss = sir_loss(predictions, SIR_tensor, t_data, data['population'].iloc[0])
        # reg_loss = model.regularization()  # Un-comment and compute regularization loss
        # total_loss = loss + reg_loss  # Combine primary loss and regularization loss
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update the learning rate according to the scheduler

        history.append(
            loss.item()
        )  # Log the total loss, including regularization
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]}"
            )  # Print the current learning rate

        early_stopping(loss.item())  # Pass the total loss to early stopping
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model, history


input_dimension = 1
output_dimension = 3
n_hidden_layers = 4
neurons = 65
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
model, history = train_PINN(
    my_network, t_data, SIR_tensor, num_epochs=100000, lr=0.001
)

# Plot training history
plt.figure(figsize=(10, 4))
plt.grid(True, which="both", ls=":")
plt.plot(np.arange(1, len(history) + 1), np.log10(history), label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Log10(Loss)")
plt.legend()
plt.show()


# Evaluate and visualize model predictions
model.eval()
with torch.no_grad():
    predictions = model(t_data)
S_pred, I_pred, R_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2]

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(t_eval, S_sir, label="Actual S")
plt.plot(t_eval, S_pred.cpu().numpy(), label="Predicted S")
plt.xlabel("Time (days)")
plt.ylabel("S")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(t_eval, I_sir, label="Actual I")
plt.plot(t_eval, I_pred.cpu().numpy(), label="Predicted I")
plt.xlabel("Time (days)")
plt.ylabel("I")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(t_eval, R_sir, label="Actual R")
plt.plot(t_eval, R_pred.cpu().numpy(), label="Predicted R")
plt.xlabel("Time (days)")
plt.ylabel("R")
plt.legend()

plt.tight_layout()
plt.show()

# # Define the Neural Network
# class SIRNet(nn.Module):
#     def __init__(self, input_size=1, hidden_size=50, output_size=2, layers=4, activation_function='relu'):
#         super(SIRNet, self).__init__()
#         modules = []
#         for i in range(layers):
#             modules.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
#             if activation_function == 'relu':
#                 modules.append(nn.ReLU())
#             elif activation_function == 'leaky_relu':
#                 modules.append(nn.LeakyReLU())
#             elif activation_function == 'elu':
#                 modules.append(nn.ELU())
#             input_size = hidden_size  # Update input size for the next layer
#         modules.append(nn.Linear(hidden_size, output_size))
#         self.net = nn.Sequential(*modules)
    
#     def forward(self, t):
#         return self.net(t)

# # Loss Function
# def sir_loss(model_output, SIR_tensor, t, model, beta, gamma, N, regularization_factor=0.01):
#     I_pred, R_pred = model_output[:, 0], model_output[:, 1]
#     I_actual, R_actual = SIR_tensor[:, 0], SIR_tensor[:, 1]
    
#     loss_fit = torch.mean((I_pred - I_actual) ** 2) + torch.mean((R_pred - R_actual) ** 2)
    
#     # Ensure autograd is enabled for the tensors involved in the physics-based loss
#     I_t = torch.autograd.grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
#     R_t = torch.autograd.grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    
#     dIdt_pred = beta * I_pred - gamma * I_pred
#     dRdt_pred = gamma * I_pred
    
#     loss_sir = torch.mean((I_t - dIdt_pred) ** 2) + torch.mean((R_t - dRdt_pred) ** 2)
#     l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
#     loss = loss_fit + loss_sir + regularization_factor * l2_reg
    
#     return loss

# # Training Loop
# def train(model, epochs, beta, gamma, N, regularization_factor=1e-1):
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.22)
    
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         model_output = model(t_tensor)
#         loss = sir_loss(model_output, SIR_tensor, t_tensor, model, beta, gamma, N, regularization_factor)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
        
#         if epoch % 500 == 0:
#             print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

# # Initialize and Train the Model
# model = SIRNet()
# train(model, epochs=50000, beta=1.30, gamma=0.15, N=data['population'][0])

# # Switch the model to evaluation mode
# model.eval()

# # Generate predictions for the same inputs used during training
# with torch.no_grad():
#     predictions = model(t_tensor)

# # Extract the predicted I, R values
# I_pred, R_pred = predictions[:, 0].numpy(), predictions[:, 1].numpy()

# # Extract the actual I, R values from the SIR_tensor
# I_actual, R_actual = SIR_tensor[:, 0].numpy(), SIR_tensor[:, 1].numpy()

# # Extract the time points from t_tensor for plotting
# time_points = t_tensor.numpy().flatten()

# # Plotting the actual vs. predicted data
# plt.figure(figsize=(12, 8))

# plt.plot(time_points, I_actual, 'r', label='Infected Actual', linewidth=2)
# plt.plot(time_points, I_pred, 'r--', label='Infected Predicted', linewidth=2)

# plt.plot(time_points, R_actual, 'g', label='Recovered Actual', linewidth=2)
# plt.plot(time_points, R_pred, 'g--', label='Recovered Predicted', linewidth=2)

# plt.xlabel('Days Since Start')
# plt.ylabel('Population')
# plt.title('SIR Model Predictions vs. Actual Data')
# plt.legend()
# plt.grid(True)
# plt.show()
