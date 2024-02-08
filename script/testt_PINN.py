import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Parameters
N = 1000
beta = 0.5
gamma = 0.2
I0 = 1
S0 = N - I0
R0 = 0
t = np.linspace(0, 50, 1000)

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# observed_I = np.array(I)
# observed_R = np.array(R)
# t_observed = np.linspace(0, t, len(observed_I))

# Convert to torch tensors
t_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1)
S_tensor = torch.tensor(S, dtype=torch.float32).view(-1, 1)
I_tensor = torch.tensor(I, dtype=torch.float32).view(-1, 1)
R_tensor = torch.tensor(R, dtype=torch.float32).view(-1, 1)
SIR_tensor = torch.cat([S_tensor, I_tensor, R_tensor], 1)
t_tensor.requires_grad = True

# Define the neural network
class SIRNet(nn.Module):
    def __init__(self):
        super(SIRNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 3)
        )
    
    def forward(self, t):
        return self.net(t)

# Physics-informed loss
def sir_loss(model_output, SIR_tensor, t, N):
    S_pred, I_pred, R_pred = model_output[:, 0], model_output[:, 1], model_output[:, 2]
    S_t = torch.autograd.grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    I_t = torch.autograd.grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    R_t = torch.autograd.grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    
    dSdt = -beta * S_pred * I_pred / N
    dIdt = beta * S_pred * I_pred / N - gamma * I_pred
    dRdt = gamma * I_pred
    
    loss = torch.mean((S_t - dSdt) ** 2) + torch.mean((I_t - dIdt) ** 2) + torch.mean((R_t - dRdt) ** 2)
    loss += torch.mean((model_output - SIR_tensor) ** 2)  # Data fitting loss
    return loss

# Training function
def train(model, t_tensor, SIR_tensor, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        model_output = model(t_tensor)
        loss = sir_loss(model_output, SIR_tensor, t_tensor, N)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Initialize the model and train
model = SIRNet()
train(model, t_tensor, SIR_tensor, epochs=5000, lr=0.001)

# Predict and plot
model.eval()
with torch.no_grad():
    predictions = model(t_tensor)

plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
plt.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
plt.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
plt.plot(t, predictions[:, 0].numpy(), 'b--', label='Susceptible (predicted)')
plt.plot(t, predictions[:, 1].numpy(), 'y--', label='Infected (predicted)')
plt.plot(t, predictions[:, 2].numpy(), 'g--', label='Recovered (predicted)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Assumed you have your observed data for I and R loaded into observed_I and observed_R
# t_observed is also assumed to be defined correctly
N = 1000  # Total population, adjust as per your scenario

# t_tensor = torch.tensor(t_observed, dtype=torch.float32).view(-1, 1)
# I_tensor = torch.tensor(observed_I, dtype=torch.float32).view(-1, 1)
# R_tensor = torch.tensor(observed_R, dtype=torch.float32).view(-1, 1)
# t_tensor.requires_grad = True

class SIR_inversNet(nn.Module):
    def __init__(self):
        super(SIR_inversNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),  # Adjusted to take 1-dimensional input
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2)  # Outputs for I and R
        )
        
        self.beta = nn.Parameter(torch.rand(1, requires_grad=True))
        self.gamma = nn.Parameter(torch.rand(1, requires_grad=True))
    
    def forward(self, t):
        return self.net(t)
    
    
def loss_fn(model_output, I_tensor, R_tensor, t, N, model):
    I_pred, R_pred = model_output[:, 0], model_output[:, 1]
    
    # Ensure t is properly shaped for differentiation
    t.requires_grad_(True)
    
    # Compute derivatives
    I_t = torch.autograd.grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    R_t = torch.autograd.grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    
    # Using the model's parameters to compute the expected derivatives
    dIdt_pred = model.beta * (N - I_pred - R_pred) * I_pred / N - model.gamma * I_pred
    dRdt_pred = model.gamma * I_pred
    
    # Loss is the sum of the squared differences
    loss = torch.mean((I_t - dIdt_pred) ** 2) + torch.mean((R_t - dRdt_pred) ** 2)
    # Add data fitting loss for I and R
    loss += torch.mean((I_pred - I_tensor) ** 2) + torch.mean((R_pred - R_tensor) ** 2)
    return loss


def train(model, t_tensor, I_tensor, R_tensor, epochs, lr, N):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        model_output = model(t_tensor)
        loss = loss_fn(model_output, I_tensor, R_tensor, t_tensor, N, model)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

epochs = 5000
lr = 0.001

model = SIR_inversNet()
train(model, t_tensor, I_tensor, R_tensor, epochs, lr, N)

model.eval()
with torch.no_grad():
    predictions = model(t_tensor)

plt.figure(figsize=(10, 6))
plt.plot(t, I, 'y', label='Infected (observed)')
plt.plot(t, R, 'g', label='Removed (observed)')
plt.plot(t, predictions[:, 0].numpy(), 'y--', label='Infected (predicted)')
plt.plot(t, predictions[:, 1].numpy(), 'g--', label='Removed (predicted)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

print("Estimated beta:", model.beta.item())
print("Estimated gamma:", model.gamma.item())


import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Parameters
N = 1000
beta = 0.5
gamma = 0.2
I0 = 1
S0 = N - I0
R0 = 0
t = np.linspace(0, 50, 1000)

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Convert NumPy arrays to PyTorch tensors directly
t_tensor = torch.from_numpy(t).float().view(-1, 1)
S_tensor = torch.from_numpy(S).float().view(-1, 1)
I_tensor = torch.from_numpy(I).float().view(-1, 1)
R_tensor = torch.from_numpy(R).float().view(-1, 1)
SIR_tensor = torch.cat([S_tensor, I_tensor, R_tensor], dim=1)
t_tensor.requires_grad = True


class SIRNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=3, layers=4, activation_function='relu'):
        super(SIRNet, self).__init__()
        modules = []
        for i in range(layers):
            if i == 0:
                modules.append(nn.Linear(input_size, hidden_size))
            else:
                modules.append(nn.Linear(hidden_size, hidden_size))
            
            if activation_function == 'relu':
                modules.append(nn.ReLU())
            elif activation_function == 'leaky_relu':
                modules.append(nn.LeakyReLU())
            elif activation_function == 'elu':
                modules.append(nn.ELU())
        
        # Output layer
        modules.append(nn.Linear(hidden_size, output_size))
        
        self.net = nn.Sequential(*modules)
    
    def forward(self, t):
        return self.net(t)


def sir_loss(model_output, SIR_tensor, t, N, model, regularization_factor=0.01):
    S_pred, I_pred, R_pred = model_output[:, 0], model_output[:, 1], model_output[:, 2]
    S_t = torch.autograd.grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    I_t = torch.autograd.grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    R_t = torch.autograd.grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    
    dSdt = -beta * S_pred * I_pred / N
    dIdt = beta * S_pred * I_pred / N - gamma * I_pred
    dRdt = gamma * I_pred
    
    loss = torch.mean((S_t - dSdt) ** 2) + torch.mean((I_t - dIdt) ** 2) + torch.mean((R_t - dRdt) ** 2)
    loss += torch.mean((model_output - SIR_tensor) ** 2)  # Data fitting loss
    
    # L2 regularization
    l2_reg = sum(param.pow(2).sum() for param in model.parameters())
    loss += regularization_factor * l2_reg
    
    return loss

def train(model, t_tensor, SIR_tensor, epochs, lr, regularization_factor=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)  # Dynamic LR adjustment
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        model_output = model(t_tensor)
        loss = sir_loss(model_output, SIR_tensor, t_tensor, N, model, regularization_factor)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update the learning rate
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Initialize the SIRNet model with customizable parameters
model = SIRNet(input_size=1, hidden_size=50, output_size=3, layers=4, activation_function='relu')

# Call the train function with specified parameters
train(model, t_tensor, SIR_tensor, epochs=10000, lr=0.001, regularization_factor=0.01)

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predictions = model(t_tensor)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
plt.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Infected')
plt.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
plt.plot(t, predictions[:, 0].numpy(), 'b--', label='Susceptible (predicted)')
plt.plot(t, predictions[:, 1].numpy(), 'y--', label='Infected (predicted)')
plt.plot(t, predictions[:, 2].numpy(), 'g--', label='Recovered (predicted)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()


