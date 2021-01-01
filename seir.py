import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 83200000
# Initial number of exposed, infected and recovered individuals, E0, I0 and R0.
E0, I0, R0 = 0, 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 - E0
# infection rate, alpha, and mean recovery rate, gamma, (in 1/days).
# social distancing parameter rho representing the behaviour of the population
# and the measures of the politics
alpha, gamma, rho = 1/5.5, 1./3, 0.9
# basic reproduction rate, R0
R_0 = 2.4
# contact rate, beta, in 1/days
beta = R_0 * gamma
# A grid of time points (in days)
t = np.linspace(0, 365, 365)


# The SEIR model differential equations with social distancing parameter rho
def deriv(y, t, N, alpha, beta, gamma, rho):
    S, E, I, R = y
    dSdt = -rho * beta * S * I / N
    dEdt = rho * beta * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


# Initial conditions vector
y0 = S0, E0, I0, R0
# Integrate the SEIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, alpha, beta, gamma, rho))
S, E, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, E, 'y', alpha=0.5, lw=2, label='Exposed')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time / days')
ax.set_ylabel('Number')
ax.set_ylim(0, 1.05 * N)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
plt.show()
