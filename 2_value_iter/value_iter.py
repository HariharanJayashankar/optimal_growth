import sys
import numpy as np
sys.path.insert(0, '../value_iter')
from bellman_val import bellman_updater  # the bellman updater
sys.path.insert(0, '../analytic_bellman')
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
fig.savefig('./error.png')


# plotting optimal values

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(grid, w, marker='o', label="Numerical Solution")
ax.plot(grid, mdl.v_star(grid), label="Analytical Solution")
ax.legend()
ax.set_title("Bellman Value Function: Numerical Solution vs Analytical Solution")
fig.savefig('./value_comparison.png')

# plotting policy functions

fig, ax = plt.subplots(figsize = (9, 6))
ax.plot(grid, pol, marker = 'o', label = 'Numerical Solution')
ax.plot(grid, mdl.c_star(grid), label = "Analytic Solution")
ax.legend()
ax.set_title("Bellman Policy Function: Numerical Solution vs Analytical Solution")
fig.savefig('./policy_comparison.png')

# Convergence happens, although not perfectly.
