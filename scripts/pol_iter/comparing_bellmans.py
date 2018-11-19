import numpy as np
import sys
sys.path.insert(0, 'C:/Users/admin/Documents/personal/optimal_growth/scripts/analytic_bellman')
from analytic_bellman import log_consumption
sys.path.insert(0, 'C:/Users/admin/Documents/personal/optimal_growth/scripts/pol_iter')
from bellman_pol import eul_updater
import matplotlib.pyplot as plt


# load in our analytical model and assign all parameters
mdl = log_consumption()
alpha, beta, s, mu = mdl.alpha, mdl.beta, mdl.s, mdl.mu
u, u_prime, f, f_prime = mdl.u, mdl.u_prime, mdl.f, mdl.f_prime

# Using our bellman updater

grid_max = 4
grid_size = 200
shock_size = 250

grid = np.linspace(1e-5, grid_max, grid_size)
shocks = np.exp(mu + s * np.random.randn(shock_size))

#using analyic solution to see if our policy iteration works
g = eul_updater(mdl.c_star(grid),
                grid,
                u_prime,
                beta,
                f,
                f_prime,
                shocks)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(grid, g, marker = 'o', label="Bellman Update")
ax.plot(grid, mdl.c_star(grid), label="Analytical Solution")
plt.show()

#==iterating and plotting==#

g = grid #initial guessx
n = 20
fig, ax = plt.subplots(figsize=(8, 5))
lb = 'initial condition $c(y) = y$'
ax.plot(grid, g, color=plt.cm.jet(0), lw=2, alpha=0.6, label=lb)
for i in range(n):
    new_g = eul_updater(g, grid, u_prime, beta, f, f_prime, shocks)
    g = new_g
    ax.plot(grid, g, color=plt.cm.jet(i / n), lw=2, alpha=0.6)

lb = 'true policy function $c^*$'
ax.plot(grid, mdl.c_star(grid), 'k-', lw=2, alpha=0.8, label=lb)
ax.legend(loc='upper left')

plt.show()

