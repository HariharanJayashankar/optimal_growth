import sys
sys.path.insert(0, 'C:/Users/admin/Documents/personal/optimal_growth/scripts')
import numpy as np
from bellman import bellman_updater  # the bellman updater
from analytic_bellman import log_consumption
import matplotlib.pyplot as plt

# parameters
mdl = log_consumption()
alpha, beta, s, mu = mdl.alpha, mdl.beta, mdl.s, mdl.mu

# grid parameteres
grid_size = 200
grid = np.linspace(1e-5, 4, grid_size)

# shocks
shock_size = 250
shocks = np.exp(mu + s * np.random.randn(shock_size))

# initial w
w = np.empty_like(grid)

# == solving bellman == #
tol = 1e-16
iter = 5000

# Tw = np.empty_like(grid)
error = tol + 1
i = 0
errors = []

while error > tol and i < iter:
    w_1, pol = bellman_updater(w,
                               grid,
                               np.log,
                               beta,
                               lambda k: k**alpha,
                               shocks)
    error = np.max(np.abs(w_1 - w))
    w = w_1
    print('iter ' + str(i) + '\n' + 'error: ' + str(error))
    errors.append(error)
    i += 1

# plotting error over time

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(np.linspace(0, iter, num=iter), errors, label="Error")
ax.legend()
ax.set_title('Error Rate over Iteration')
fig.savefig('C:/Users/admin/Documents/personal/optimal_growth/figures/error.png')


# plotting optimal values

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(grid, w, marker='o', label="Numerical Solution")
ax.plot(grid, mdl.v_star(grid), label="Analytical Solution")
ax.legend()
ax.set_title("Bellman: Numerical Solution vs Analytical Solution")
fig.savefig('C:/Users/admin/Documents/personal/optimal_growth/figures/bellman_comparison.png')

# Convergence happens, although not perfectly. Even for 5000 iterations, the error rate oscilates around 5e-14
# for quite a while. Why does this happen? Is it not possible for us to get lower error rates?
