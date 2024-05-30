import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the differential equations
def seird_model(y, t, N, beta, alpha, rho, da, ds, omega, dH, gamma_c, delta_c, mu, eta):
    S, E, Ia, Is, H, C, R, D = y
    dSdt = -beta * S * (Is + Ia) / N + eta * R
    dEdt = beta * S * (Is + Ia) / N - alpha * E
    dIadt = (1 - rho) * alpha * E - Ia / da
    dIsdt = rho * alpha * E - Is / ds
    dHdt = omega * Is / ds - H / dH - mu * H
    dCdt = (1 - omega) * H / dH - gamma_c * C - delta_c * C
    dRdt = Ia / da + H / dH + gamma_c * C - eta * R
    dDdt = mu * H + delta_c * C
    return [dSdt, dEdt, dIadt, dIsdt, dHdt, dCdt, dRdt, dDdt]

# Total population, N
N = 1000

# Initial number of infected and recovered individuals, everyone else is susceptible
E0, Ia0, Is0, H0, C0, R0, D0 = 1, 0, 0, 0, 0, 0, 0
S0 = N - E0 - Ia0 - Is0 - H0 - C0 - R0 - D0

# Contact rate, beta; incubation period, alpha; proportion of symptomatic, rho
beta = 0.5
alpha = 1 / 5.0
rho = 0.75

# Infectious periods
da = 10
ds = 5
dH = 14

# Hospitalisation rate, omega; recovery rate from critical care, gamma_c; death rates
omega = 0.2
gamma_c = 0.05
delta_c = 0.02
mu = 0.05
eta = 0.01

# Initial conditions vector
y0 = [S0, E0, Ia0, Is0, H0, C0, R0, D0]

# Time grid (in days)
t = np.linspace(0, 160, 160)

# Integrate the SEIRD equations over the time grid, t
ret = odeint(seird_model, y0, t, args=(N, beta, alpha, rho, da, ds, omega, dH, gamma_c, delta_c, mu, eta))
S, E, Ia, Is, H, C, R, D = ret.T

# Plot the data
fig = plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, E, 'y', label='Exposed')
plt.plot(t, Ia, 'g', label='Infected Asymptomatic')
plt.plot(t, Is, 'r', label='Infected Symptomatic')
plt.plot(t, H, 'purple', label='Hospitalised')
plt.plot(t, C, 'orange', label='Critical')
plt.plot(t, R, 'c', label='Recovered')
plt.plot(t, D, 'k', label='Deceased')
plt.xlabel('Time /days')
plt.ylabel('Number')
plt.title('SEIRD Model Simulation')
plt.legend(loc='best')
plt.grid(True)
plt.show()
