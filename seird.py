import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Total population, N.
N = 83200000
# Initial number of exposed, infected, recovered and dead individuals,
# E0, I0, R0 and D0.
E0, I0, R0, D0, C0 = 4000, 1000, 0, 0, 1000
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 - E0 - D0
# basic reproduction rate, R0
R0 = 2.4
# infection rate, alpha, contact rate, beta, mean recovery rate, gamma,
# and mortality rate, mu.(in 1/days).
alpha, gamma, mu = 1/5.5, 1./3, 0.001
beta = R0 * gamma
# A grid of time points (in days)
t = np.linspace(0, 365, 365)


# The SEIRD model differential equations.
def deriv(y, t, N, alpha, beta, gamma, mu):
    S, E, I, R, D, C = y
    # a time dependent social distancing factor, rho, representing the
    # behaviour of the population and the measures of the politics
    if t < 50:
        rho = 1
    elif 51 <= t <= 81:
        rho = 0.6
    else:
        rho = 0.2
    dSdt = -rho * beta * S * I / N
    dEdt = rho * beta * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    dCdt = rho * beta * S * I / N
    return dSdt, dEdt, dIdt, dRdt, dDdt, dCdt


# Initial conditions vector
y0 = S0, E0, I0, R0, D0, C0
# Integrate the SEIRD equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, alpha, beta, gamma, mu))
S, E, I, R, D, C = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, E, 'y', alpha=0.5, lw=2, label='Exposed')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.plot(t, D, 'black', alpha=0.5, lw=2, label='Dead')
ax.plot(t, C, 'orange', alpha=0.5, lw=2, label='Cumulative Cases')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number')
legend = ax.legend()
ax.set_ylim(0, 1.05 * N)
legend.get_frame().set_alpha(0.5)
plt.show()
