import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 83200000
# Initial number of actual infected, I0, cumulative infected, C0, recovered and
# dead individuals, I0, C0, R0 and D0.
I0, R0, D0, C0 = 1, 0, 0, 1
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 - D0
# Mean recovery rate, gamma, and mortality rate, mu, (in 1/days).
gamma, mu = 1/3, 0.001
# basic reproduction rate, R_0
R_0 = 2.4
# Contact rate, beta, in 1/days
beta = R_0 * gamma
# A grid of time points (in days)
t = np.linspace(0, 365, 365)


# The SIRD model differential equations.
def deriv(y, t, N, beta, gamma, mu):
    S, I, R, D, C = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I - mu * I
    dRdt = gamma * I
    dDdt = mu * I
    dCdt = beta * S * I / N
    return dSdt, dIdt, dRdt, dDdt, dCdt


# Initial conditions vector
y0 = S0, I0, R0, D0, C0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, mu))
S, I, R, D, C = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.plot(t, D, 'black', alpha=0.5, lw=2, label='Dead')
ax.plot(t, C, 'orange', alpha=0.5, lw=2, label='Cumulative')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number')
ax.set_ylim(0, 1.05 * N)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
plt.show()
