import sys
import numpy as np
sys.path.insert(0, '../analytic_bellman')
from analytic_bellman import log_consumption
sys.path.insert(0, '../value_iter')
from bellman import bellman_updater
import matplotlib.pyplot as plt

# load in our analytical model and assign all parameters
mdl = log_consumption()
alpha, beta, s, mu = mdl.alpha, mdl.beta, mdl.s, mdl.mu

# Using our bellman updater

grid_max = 4
grid_size = 200
shock_size = 250

grid = np.linspace(1e-5, grid_max, grid_size)
shocks = np.exp(mu + s * np.random.randn(shock_size))

# inserting v_star into our bellman_updater

w_star = bellman_updater(mdl.v_star(grid),
                         grid,
                         np.log,
                         beta,
                         lambda k: k**alpha,
                         shocks)[0]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(grid, w_star, marker='o', label="Bellman Update")
ax.plot(grid, mdl.v_star(grid), label="Analytical Solution")
ax.legend(loc="lower right")
plt.show()

# ==comparing solutions after iterating a bit== #

w = 5 * np.log(grid)  # An initial condition
n = 500
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(np.min(grid), np.max(grid))
lb = 'initial condition'
ax.plot(grid, w, color=plt.cm.jet(0), lw=2, alpha=0.6, label=lb)
for i in range(n):
    w = bellman_updater(w,
                        grid,
                        np.log,
                        beta,
                        lambda k: k**alpha,
                        shocks)[0]

lb = 'true value function'
ax.plot(grid, mdl.v_star(grid), 'k-', lw=2, alpha=0.8, label=lb)
ax.legend(loc='lower right')
plt.show()

# This seems to be working. We seem to converge to our true solution
# So our code seems ok. It seems that we are updating our bellman correctly
