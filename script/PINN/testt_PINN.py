import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.style.use("fivethirtyeight")
plt.rcParams.update({
    "lines.linewidth": 2,
    "font.family": "serif",
    "axes.titlesize": 20,
    "axes.labelsize": 14,
    "figure.figsize": [15, 8],
    "figure.autolayout": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.75",
    "legend.fontsize": "medium"
})

def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

N = 1000  # Total population
beta = 0.5  # Infection rate
gamma = 0.2  # Recovery rate
I0, R0 = 1, 0  # Initial number of infected and recovered individuals
S0 = N - I0  # Initial number of susceptible individuals
t = np.linspace(0, 50, 1000)  # Time grid

y0 = S0, I0, R0
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

t_tensor = torch.tensor(t, dtype=torch.float32, device=device).view(-1, 1)
S_tensor = torch.tensor(S, dtype=torch.float32, device=device).view(-1, 1)
I_tensor = torch.tensor(I, dtype=torch.float32, device=device).view(-1, 1)
R_tensor = torch.tensor(R, dtype=torch.float32, device=device).view(-1, 1)
SIR_tensor = torch.cat([S_tensor, I_tensor, R_tensor], dim=1)
t_tensor.requires_grad = True

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

model = SIRNet().to(device)

def sir_loss(model_output, SIR_tensor, t, N, beta, gamma):
    S_pred, I_pred, R_pred = model_output[:, 0], model_output[:, 1], model_output[:, 2]
    S_t = torch.autograd.grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    I_t = torch.autograd.grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    R_t = torch.autograd.grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]

    dSdt = -(beta * S_pred * I_pred) / N
    dIdt = (beta * S_pred * I_pred) / N - gamma * I_pred
    dRdt = gamma * I_pred

    loss = torch.mean((S_t - dSdt) ** 2) + torch.mean((I_t - dIdt) ** 2) + torch.mean((R_t - dRdt) ** 2)
    loss += torch.mean((model_output - SIR_tensor) ** 2)  # Data fitting loss
    return loss

def train(model, t_tensor, SIR_tensor, epochs, lr, N, beta, gamma):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        model_output = model(t_tensor)
        loss = sir_loss(model_output, SIR_tensor, t_tensor, N, beta, gamma)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

train(model, t_tensor, SIR_tensor, epochs=5000, lr=0.001, N=N, beta=beta, gamma=gamma)

model.eval()
with torch.no_grad():
    predictions = model(t_tensor).cpu()

plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'y', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.plot(t, predictions[:, 0], 'b--', label='Susceptible (predicted)')
plt.plot(t, predictions[:, 1], 'y--', label='Infected (predicted)')
plt.plot(t, predictions[:, 2], 'g--', label='Recovered (predicted)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

# Inverse problem

class SIRInverseNet(nn.Module):
    def __init__(self):
        super(SIRInverseNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 2)  # Still outputting only I and R predictions
        )
        self.beta = nn.Parameter(torch.rand(1, device=device, requires_grad=True))
        self.gamma = nn.Parameter(torch.rand(1, device=device, requires_grad=True))

    def forward(self, t):
        ir = self.net(t)
        I_pred = ir[:, 0]
        R_pred = ir[:, 1]
        S_pred = N - I_pred - R_pred  # Compute S from I and R
        return S_pred, I_pred, R_pred

inverse_model = SIRInverseNet().to(device)

def inverse_loss(S_pred, I_pred, R_pred, S_tensor, I_tensor, R_tensor, t, N, model, initial_conditions_weight=1.0):
    # Compute the derivatives of S, I, and R
    S_t = torch.autograd.grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    I_t = torch.autograd.grad(I_pred, t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    R_t = torch.autograd.grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]

    # SIR model equations
    dSdt = -model.beta * S_pred * I_pred / N
    dIdt = model.beta * S_pred * I_pred / N - model.gamma * I_pred
    dRdt = model.gamma * I_pred

    # Loss based on the difference between predicted derivatives and expected derivatives
    loss = torch.mean((S_t - dSdt) ** 2) + torch.mean((I_t - dIdt) ** 2) + torch.mean((R_t - dRdt) ** 2)
    
    # Loss based on the differences between the predicted and observed values
    loss += torch.mean((S_pred - S_tensor) ** 2) + torch.mean((I_pred - I_tensor) ** 2) + torch.mean((R_pred - R_tensor) ** 2)

    # Penalize deviations from the initial conditions
    initial_conditions_loss = initial_conditions_weight * (
        (S_pred[0] - S_tensor[0]) ** 2 +
        (I_pred[0] - I_tensor[0]) ** 2 +
        (R_pred[0] - R_tensor[0]) ** 2
    )

    return loss + initial_conditions_loss

def inverse_train(model, t_tensor, S_tensor, I_tensor, R_tensor, epochs, lr, N, initial_conditions_weight=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    beta_values = []
    gamma_values = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        S_pred, I_pred, R_pred = model(t_tensor)
        loss = inverse_loss(S_pred, I_pred, R_pred, S_tensor, I_tensor, R_tensor, t_tensor, N, model, initial_conditions_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Record the parameters
        beta_values.append(model.beta.item())
        gamma_values.append(model.gamma.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}, Beta: {model.beta.item()}, Gamma: {model.gamma.item()}")

    return beta_values, gamma_values



observed_I = I  # This should be your actual observed data for Infected
observed_R = R  # This should be your actual observed data for Recovered
t_observed = t
# Compute observed S data
observed_S = N - observed_I - observed_R
S_tensor = torch.tensor(observed_S, dtype=torch.float32, device=device).view(-1, 1)
I_tensor = torch.tensor(observed_I, dtype=torch.float32, device=device).view(-1, 1)
R_tensor = torch.tensor(observed_R, dtype=torch.float32, device=device).view(-1, 1)

beta_vals, gamma_vals = inverse_train(inverse_model, t_tensor, S_tensor, I_tensor, R_tensor, epochs=50000, lr=0.001, N=N)


inverse_model.eval()
with torch.no_grad():
    S_pred, I_pred, R_pred = inverse_model(t_tensor)
    S_pred = S_pred.cpu()
    I_pred = I_pred.cpu()
    R_pred = R_pred.cpu()

# Now you can plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(t_observed, observed_I, 'y', label='Infected (observed)')
plt.plot(t_observed, observed_R, 'g', label='Recovered (observed)')
plt.plot(t_observed, I_pred, 'y--', label='Infected (predicted)')
plt.plot(t_observed, R_pred, 'g--', label='Recovered (predicted)')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

print(f"Estimated beta: {inverse_model.beta.item()}")
print(f"Estimated gamma: {inverse_model.gamma.item()}")


# Beta values plot
plt.subplot(1, 2, 1)
plt.plot(beta_vals, label='Beta values')
plt.xlabel('Epoch')
plt.ylabel('Beta')
plt.title('Beta Parameter over Epochs')
plt.legend()

# Gamma values plot
plt.subplot(1, 2, 2)
plt.plot(gamma_vals, label='Gamma values')
plt.xlabel('Epoch')
plt.ylabel('Gamma')
plt.title('Gamma Parameter over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
