from scipy.integrate import solve_ivp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SAIRD Model Parameters and Initial Conditions as per the paper
params = {
    "rho1": 0.80,
    "rho2": 0.29,
    "alpha": 0.1,
    "beta": 0.17,
    "gamma": 1 / 16,
    "theta": 0.001,
    "N": 1000,
}
initial_conditions = [970, 10, 20, 0, 0]  # [S0, A0, I0, R0, D0]


# Define the SAIRD model differential equations
def saird_model(t, y, params):
    S, A, I, R, D = y
    N = params["N"]
    dSdt = -params["beta"] * I * S / N - params["alpha"] * A * S / N
    dAdt = (
        params["rho2"] * params["beta"] * I * S / N
        + (1 - params["rho1"]) * params["alpha"] * A * S / N
        - params["gamma"] * A
        - params["theta"] * A
    )
    dIdt = (
        (1 - params["rho2"]) * params["beta"] * I * S / N
        + params["rho1"] * params["alpha"] * A * S / N
        - params["gamma"] * I
        - params["theta"] * I
    )
    dRdt = params["gamma"] * (I + A)
    dDdt = params["theta"] * (I + A)
    return [dSdt, dAdt, dIdt, dRdt, dDdt]


# Generate synthetic SAIRD data
t_span = [0, 100]  # 100 days
t_eval = np.linspace(t_span[0], t_span[1], 100)  # 100 data points
saird_solution = solve_ivp(
    saird_model,
    t_span,
    initial_conditions,
    args=(params,),
    t_eval=t_eval,
    method="RK45",
)

# Extract SIR data from SAIRD solution
S_saird, A_saird, I_saird, R_saird, D_saird = saird_solution.y
S_sir = S_saird + A_saird  # S compartment for SIR
R_sir = R_saird + D_saird  # R compartment for SIR
I_sir = I_saird.copy()  # I compartment is the same in both models

t_data = torch.tensor(t_eval, dtype=torch.float32).reshape(-1, 1).to(device)
S_data = torch.tensor(S_sir, dtype=torch.float32).reshape(-1, 1).to(device)
I_data = torch.tensor(I_sir, dtype=torch.float32).reshape(-1, 1).to(device)
R_data = torch.tensor(R_sir, dtype=torch.float32).reshape(-1, 1).to(device)
SIR_tensor = torch.cat([S_data, I_data, R_data], 1)
t_data.requires_grad = True

# class NeuralNet(nn.Module):
#     def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, regularization_param, regularization_exp, retrain_seed):
#         super(NeuralNet, self).__init__()
#         self.input_dimension = input_dimension
#         self.output_dimension = output_dimension
#         self.neurons = neurons
#         self.n_hidden_layers = n_hidden_layers
#         self.activation = nn.Tanh()
#         self.regularization_param = regularization_param
#         self.regularization_exp = regularization_exp
#         self.retrain_seed = retrain_seed

#                 # Learnable SIR parameters, initialized to log of initial guesses to ensure positivity
#         self.log_beta = nn.Parameter(torch.log(torch.tensor([0.25], device=device)))
#         self.log_gamma = nn.Parameter(torch.log(torch.tensor([0.15], device=device)))

#         self.input_layer = nn.Linear(self.input_dimension, self.neurons)
#         self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
#         self.output_layer = nn.Linear(self.neurons, self.output_dimension)

#         self.init_xavier()


#     def forward(self, x):
#         x = self.activation(self.input_layer(x))
#         for layer in self.hidden_layers:
#             x = self.activation(layer(x))
#         return self.output_layer(x)

#     @property
#     def beta(self):
#         return torch.exp(self.log_beta)

#     @property
#     def gamma(self):
#         return torch.exp(self.log_gamma)

#     def init_xavier(self):
#         torch.manual_seed(self.retrain_seed)

#         def init_weights(m):
#             if isinstance(m, nn.Linear):
#                 g = nn.init.calculate_gain('tanh')
#                 nn.init.xavier_uniform_(m.weight, gain=g)
#                 m.bias.data.fill_(0)

#         self.apply(init_weights)

#     def regularization(self):
#         reg_loss = 0
#         for name, param in self.named_parameters():
#             if 'weight' in name:
#                 reg_loss += torch.norm(param, self.regularization_exp)
#         return self.regularization_param * reg_loss


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


# def sir_loss(model, SIR_tensor, t, N):
#     model_output = model(t)
#     S_pred, I_pred, R_pred = model_output[:, 0], model_output[:, 1], model_output[:, 2]

#     S_t = torch.autograd.grad(S_pred.sum(), t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
#     I_t = torch.autograd.grad(I_pred.sum(), t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
#     R_t = torch.autograd.grad(R_pred.sum(), t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]

#     dSdt = -model.beta * S_pred * I_pred / N
#     dIdt = model.beta * S_pred * I_pred / N - model.gamma * I_pred
#     dRdt = model.gamma * I_pred

#     loss = torch.mean((S_t - dSdt)**2) + torch.mean((I_t - dIdt)**2) + torch.mean((R_t - dRdt)**2)
#     loss += torch.mean((model_output - SIR_tensor)**2)


#     return loss
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


# def train_PINN(model, t_data, SIR_tensor, num_epochs=5000, lr=0.01):
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     early_stopping = EarlyStopping(patience=300, verbose=True)
#     history = []

#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         predictions = model(t_data)
#         loss = sir_loss(predictions, SIR_tensor, t_data, params['N'])
#         # reg_loss = model.regularization()
#         # total_loss = loss + reg_loss
#         loss.backward()
#         optimizer.step()

#         history.append(loss.item())
#         if (epoch + 1) % 500 == 0 or epoch == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

#         early_stopping(loss)
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break


#     return model, history
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
        loss = sir_loss(predictions, SIR_tensor, t_data, params["N"])
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
