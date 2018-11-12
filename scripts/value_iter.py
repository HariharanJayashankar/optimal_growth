import sys
sys.path.insert(0, 'C:/Users/admin/Documents/personal/optimal_growth/scripts')
import numpy as np
from bellman import bellman_updater #the bellman updater
from analytic_bellman import log_consumption
import matplotlib.pyplot as plt

#parameters
mdl = log_consumption()
alpha, beta, s, mu = mdl.alpha, mdl.beta, mdl.s, mdl.mu

#grid parameteres
grid_size = 200
grid = np.linspace(1e-5, 4, grid_size)

#shocks
shock_size = 250
shocks = np.exp(mu + s * np.random.randn(shock_size))

#initial w
w_0 = np.empty_like(grid)

## == solving bellman == ##
tol = 1e-16
iter = 500

w = w_0
Tw = np.empty_like(grid)
error = tol + 1
i = 0

while error > tol and i < iter:
    w_1, pol = bellman_updater(w,
                        grid,
                        np.log,
                        beta,
                        lambda k: k**alpha,
                        shocks,
                        Tw)

    error = np.max(np.abs(w_1 - w))
    w = w_1
    i += 1




#plotting optimal values

fig, ax = plt.subplots(figsize = (8, 5))
ax.plot(grid, w, marker = 'o', label = "Numerical Solution")
ax.plot(grid, mdl.v_star(grid), label = "Analytical Solution")
ax.legend()
plt.show()
