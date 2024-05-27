import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameters
beta = 0.3    # Example value, needs estimation
gamma = 1/10  # Recovery rate
rho = 1/5     # Hospitalization rate
delta = 0.35  # Death rate outside ICU
eta = 0.05    # Progression to critical care
kappa = 1/10  # Hospitalization recovery rate
mu = 1/13.1   # ICU recovery rate
xi = 0.22     # ICU death rate

# Initial conditions
N = 1000000   # Total population
I0 = 1        # Initial number of infectious individuals
H0 = 0        # Initial number of hospitalized individuals
C0 = 0        # Initial number of critical care individuals
R0 = 0        # Initial number of recovered individuals
D0 = 0        # Initial number of deceased individuals
S0 = N - I0   # Initial number of susceptible individuals

# Differential equations
def deriv(y, t, N, beta, gamma, rho, delta, eta, kappa, mu, xi):
    S, I, H, C, R, D = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - (gamma + rho + delta) * I
    dHdt = rho * I - (eta + kappa) * H
    dCdt = eta * H - (mu + xi) * C
    dRdt = gamma * I + mu * C
    dDdt = delta * I + xi * C
    return dSdt, dIdt, dHdt, dCdt, dRdt, dDdt

# Initial conditions vector
y0 = S0, I0, H0, C0, R0, D0

# Time points (days)
t = np.linspace(0, 160, 160)

# Solve ODE
ret = odeint(deriv, y0, t, args=(N, beta, gamma, rho, delta, eta, kappa, mu, xi))
S, I, H, C, R, D = ret.T

# Plot results
fig = plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infectious')
plt.plot(t, H, 'orange', label='Hospitalised')
plt.plot(t, C, 'purple', label='Critical')
plt.plot(t, R, 'g', label='Recovered')
plt.plot(t, D, 'k', label='Deceased')
plt.xlabel('Time (days)')
plt.ylabel('Number of people')
plt.legend()
plt.title('SIHCRD Model Dynamics')
plt.grid(True)
plt.show()


# Parameters
beta = 0.3    # Example value, needs estimation
sigma = 1/5   # Incubation rate
gamma = 1/10  # Recovery rate
rho = 1/5     # Hospitalization rate
delta = 0.35  # Death rate outside ICU
eta = 0.05 * 1/6 # Progression to critical care
kappa = 1/10  # Hospitalization recovery rate
mu = 1/13.1   # ICU recovery rate
xi = 0.22     # ICU death rate

# Initial conditions
N = 1000   # Total population
E0 = 10       # Initial number of exposed individuals
I0 = 1        # Initial number of infectious individuals
H0 = 0        # Initial number of hospitalized individuals
C0 = 0        # Initial number of critical care individuals
R0 = 0        # Initial number of recovered individuals
D0 = 0        # Initial number of deceased individuals
S0 = N - E0 - I0  # Initial number of susceptible individuals

# Differential equations
def deriv(y, t, N, beta, sigma, gamma, rho, delta, eta, kappa, mu, xi):
    S, E, I, H, C, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - (gamma + rho + delta) * I
    dHdt = rho * I - (eta + kappa) * H
    dCdt = eta * H - (mu + xi) * C
    dRdt = gamma * I + mu * C
    dDdt = delta * I + xi * C
    return dSdt, dEdt, dIdt, dHdt, dCdt, dRdt, dDdt


# Initial conditions vector
y0 = S0, E0, I0, H0, C0, R0, D0

# Time points (days)
t = np.linspace(0, 160, 160)

# Solve ODE
ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma, rho, delta, eta, kappa, mu, xi))
S, E, I, H, C, R, D = ret.T

# Plot results
fig = plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, E, 'y', label='Exposed')
plt.plot(t, I, 'r', label='Infectious')
plt.plot(t, H, 'orange', label='Hospitalised')
plt.plot(t, C, 'purple', label='Critical')
plt.plot(t, R, 'g', label='Recovered')
plt.plot(t, D, 'k', label='Deceased')
plt.xlabel('Time (days)')
plt.ylabel('Number of people')
plt.legend()
plt.title('SEIHCRD Model Dynamics')
plt.grid(True)
plt.show()
