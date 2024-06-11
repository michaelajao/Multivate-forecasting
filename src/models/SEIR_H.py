from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# SAIRD Model Parameters and Initial Conditions as per the paper
params = {
    "rho1": 0.80,  # Probability of going from exposed to infected requiring hospitalization
    "rho2": 0.29,  # Probability of going from exposed to undetected infected
    "alpha": 0.1,  # Rate of transmission from the undetected infected
    "beta": 0.17,  # Rate of transmission from the infected requiring hospitalization
    "gamma": 1 / 16,  # Recovery rate
    "theta": 0.001,  # Mortality rate
    "N": 1000,  # Total population
    # Additional parameters for hospitalized and critical
    "rho": 0.05,  # Hospitalization rate for infected
    "delta": 0.01,  # Mortality rate for non-hospitalized
    "eta": 0.02,  # Rate at which hospitalized become critical
    "kappa": 0.03,  # Recovery rate for hospitalized
    "mu": 0.01,  # Recovery rate for critical
    "xi": 0.005  # Mortality rate for critical
}

initial_conditions = [970, 10, 20, 0, 0, 0, 0]  # [S0, E0, I0, H0, C0, R0, D0]

# Define the SEIRDC model differential equations
def seirdc_model(t, y, params):
    S, E, I, H, C, R, D = y
    N = params["N"]
    dSdt = -((params["beta"] * I + params["alpha"] * E) / N) * S
    dEdt = ((params["beta"] * I + params["alpha"] * E) / N) * S - params["gamma"] * E
    dIdt = params["rho1"] * params["gamma"] * E - params["rho"] * I - params["delta"] * I
    dHdt = params["rho"] * I - params["eta"] * H - params["kappa"] * H
    dCdt = params["eta"] * H - params["mu"] * C - params["xi"] * C
    dRdt = (1 - params["rho1"]) * params["gamma"] * E + params["kappa"] * H + params["mu"] * C
    dDdt = params["delta"] * I + params["xi"] * C
    return dSdt, dEdt, dIdt, dHdt, dCdt, dRdt, dDdt

# Time points (in days)
t_span = (0, 160)  # 100 days
t_eval = np.linspace(t_span[0], t_span[1], 160)

# Integrate the SEIRDC equations over the time grid, t.
solution = solve_ivp(seirdc_model, t_span, initial_conditions, args=(params,), t_eval=t_eval)

# Plot the data on separate curves for each compartment
plt.figure(figsize=(12, 8))
for i, label in enumerate(['Susceptible', 'Exposed', 'Infected', 'Hospitalized', 'Critical', 'Recovered', 'Deceased']):
    plt.plot(solution.t, solution.y[i], label=label)

plt.xlabel('Time /days')
plt.ylabel('Number of individuals')
plt.legend()
plt.title('SEIRDC Model Simulation')
plt.savefig(f"../../reports/figures/seirdc_model_simulation.png")
plt.show()


def oddNumbers(l, r):
    # Generate a list of odd numbers between l and r (inclusive)
    result = []
    for num in range(l, r+1):
        if num % 2 != 0:
            result.append(num)
    return result

# Testing the function with the sample input provided.
sample_input_l = 2
sample_input_r = 5
odd_numbers_result = oddNumbers(sample_input_l, sample_input_r)
odd_numbers_result