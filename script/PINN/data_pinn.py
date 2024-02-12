import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# def preprocess_data(file_path):
#     """
#     Preprocess the data for SIRD model training.

#     Parameters:
#     - file_path: str, the path to the CSV file containing the data.

#     Returns:
#     - data: pd.DataFrame, the preprocessed data with normalized columns for SIRD model.
#     """

#     # Load the data
#     data = pd.read_csv(file_path)

#     # Convert 'date' to datetime if not already in that format
#     if not pd.api.types.is_datetime64_any_dtype(data['date']):
#         data['date'] = pd.to_datetime(data['date'])

#     # Calculate the number of days since the start of the dataset
#     data['day'] = (data['date'] - data['date'].min()).dt.days

#     # # Normalize the cumulative confirmed, recovered, and deceased columns
#     # data['normalized_confirmed'] = data['cumulative_confirmed'] / data['cumulative_confirmed'].max()
#     # data['normalized_deceased'] = data['cumulative_deceased'] / data['cumulative_deceased'].max()

#     # Assuming the 'recovered' cases need to be calculated as described previously
#     recovery_period = 21  # The approximate recovery period
#     data['recovered'] = data['cumulative_confirmed'].shift(recovery_period) - data['cumulative_deceased'].shift(recovery_period)
#     # data['normalized_recovered'] = data['recovered'] / data['recovered'].max()

#     # Assuming the 'active' cases need to be calculated as described previously
#     data['active'] = data['cumulative_confirmed'] - data['cumulative_deceased'] - data['recovered']
#     # data['normalized_active'] = data['active'] / data['active'].max()

#     # Fill NaN values that may result from shifting with 0
#     data = data.fillna(0)

#     return data

# Load and Preprocess Data
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    df['new_confirmed'] = df['new_confirmed'].rolling(window=7, min_periods=1).mean().fillna(0).astype(int)
    df['new_deceased'] = df['new_deceased'].rolling(window=7, min_periods=1).mean().fillna(0).astype(int)
    recovery_period = 14
    df['recovered'] = df['cumulative_confirmed'].shift(recovery_period) - df['cumulative_deceased'].shift(recovery_period)
    df['active_cases'] = df['cumulative_confirmed'] - df['recovered'] - df['cumulative_deceased'].fillna(0)
    df['S(t)'] = df['population'] - df['cumulative_confirmed'] - df['recovered'] - df['cumulative_deceased']
    df.fillna(0, inplace=True)
    return df


df = preprocess_data("../../data/region_daily_data/East Midlands.csv")

plt.figure(figsize=(15, 10))
plt.plot(df['day'], df['active'], label='Cumulative Confirmed')
plt.show()



df = load_and_preprocess_data("../../data/region_daily_data/East Midlands.csv") 
data = df.head(30)

plt.figure(figsize=(15, 10))
plt.plot(df['days_since_start'], df['active_cases'], label='Cumulative Confirmed')
plt.show()

# Correct your file path
# data = df[df["S(t)"] > 0].head(30)  # Select first 30 data points

# Convert columns to tensors

t_tensor = torch.tensor(data['days_since_start'].values, dtype=torch.float32).view(-1, 1)
I_tensor = torch.tensor(data['cumulative_confirmed'].values, dtype=torch.float32).view(-1, 1)
R_tensor = torch.tensor(data['cumulative_deceased'].values, dtype=torch.float32).view(-1, 1)
SIR_tensor = torch.cat([I_tensor, R_tensor], dim=1)
t_tensor.requires_grad = True
# days_since_start_tensor = torch.tensor(data['days_since_start'].values, dtype=torch.float32).view(-1, 1)
# R_tensor = torch.tensor(data['R(t)'].values, dtype=torch.float32).view(-1, 1)
# I_tensor = torch.tensor(data['I(t)'].values, dtype=torch.float32).view(-1, 1)
# S_tensor = torch.tensor(data['S(t)'].values, dtype=torch.float32).view(-1, 1)
# SIR_tensor = torch.cat([S_tensor, I_tensor, R_tensor], dim=1)
# days_since_start_tensor.requires_grad = True

# Define the Neural Network
class SIRNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=2, layers=4, activation_function='relu'):
        super(SIRNet, self).__init__()
        modules = []
        for i in range(layers):
            modules.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            if activation_function == 'relu':
                modules.append(nn.ReLU())
            elif activation_function == 'leaky_relu':
                modules.append(nn.LeakyReLU())
            elif activation_function == 'elu':
                modules.append(nn.ELU())
            input_size = hidden_size  # Update input size for the next layer
        modules.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*modules)
    
    def forward(self, t):
        return self.net(t)

# Loss Function
def sir_loss(model_output, SIR_tensor, t, model, beta, gamma, N, regularization_factor=0.01):
    I_pred, R_pred = model_output[:, 0], model_output[:, 1]
    I_actual, R_actual = SIR_tensor[:, 0], SIR_tensor[:, 1]
    
    loss_fit = torch.mean((I_pred - I_actual) ** 2) + torch.mean((R_pred - R_actual) ** 2)
    
    # Ensure autograd is enabled for the tensors involved in the physics-based loss
    I_t = torch.autograd.grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    R_t = torch.autograd.grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    
    dIdt_pred = beta * I_pred - gamma * I_pred
    dRdt_pred = gamma * I_pred
    
    loss_sir = torch.mean((I_t - dIdt_pred) ** 2) + torch.mean((R_t - dRdt_pred) ** 2)
    l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
    loss = loss_fit + loss_sir + regularization_factor * l2_reg
    
    return loss

# Training Loop
def train(model, epochs, beta, gamma, N, regularization_factor=1e-):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.22)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        model_output = model(t_tensor)
        loss = sir_loss(model_output, SIR_tensor, t_tensor, model, beta, gamma, N, regularization_factor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 500 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

# Initialize and Train the Model
model = SIRNet()
train(model, epochs=50000, beta=1.30, gamma=0.15, N=data['population'][0])

# Switch the model to evaluation mode
model.eval()

# Generate predictions for the same inputs used during training
with torch.no_grad():
    predictions = model(t_tensor)

# Extract the predicted I, R values
I_pred, R_pred = predictions[:, 0].numpy(), predictions[:, 1].numpy()

# Extract the actual I, R values from the SIR_tensor
I_actual, R_actual = SIR_tensor[:, 0].numpy(), SIR_tensor[:, 1].numpy()

# Extract the time points from t_tensor for plotting
time_points = t_tensor.numpy().flatten()

# Plotting the actual vs. predicted data
plt.figure(figsize=(12, 8))

plt.plot(time_points, I_actual, 'r', label='Infected Actual', linewidth=2)
plt.plot(time_points, I_pred, 'r--', label='Infected Predicted', linewidth=2)

plt.plot(time_points, R_actual, 'g', label='Recovered Actual', linewidth=2)
plt.plot(time_points, R_pred, 'g--', label='Recovered Predicted', linewidth=2)

plt.xlabel('Days Since Start')
plt.ylabel('Population')
plt.title('SIR Model Predictions vs. Actual Data')
plt.legend()
plt.grid(True)
plt.show()
