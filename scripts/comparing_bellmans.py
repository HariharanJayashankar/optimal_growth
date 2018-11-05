import sys
sys.path.insert(0, 'C:/Users/Hariharan/Documents/projects/optimal_growth/scripts')
import numpy as np
from analytic_bellman import log_consumption
from bellman import bellman_updater
import matplotlib.pyplot as plt

mdl = log_consumption()
alpha, beta, s, mu = mdl.alpha, mdl.beta, mdl.s, mdl.mu

#Using our bellman updater

grid_max = 4
grid_size = 200
shock_size = 250

grid = np.linspace(1e-5, grid_max, grid_size)
shocks = np.exp(mu + s * np.random.randn(shock_size))

#inserting v_star into our bellman_updater

w_star = bellman_updater(mdl.v_star(grid),
                        grid,
                        np.log,
                        beta,
                        lambda k: k**alpha,
                        shocks)[0]

fig, ax = plt.subplots(figsize = (8, 5))
ax.plot(grid, w_star, marker='o', label = "Bellman Update")
ax.plot(grid, mdl.v_star(grid), label = "Analytical Solution")
ax.legend(loc = "lower right")
plt.show()
