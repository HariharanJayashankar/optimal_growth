import numpy as np
import sys
sys.path.insert(0, '../pol_iter')
from bellman_pol import eul_updater
sys.path.insert(0, '../analytic_bellman')
from analytic_bellman import log_consumption
import matplotlib.pyplot as plt


# parameters
mdl = log_consumption()
alpha, beta, s, mu = mdl.alpha, mdl.beta, mdl.s, mdl.mu
u, u_prime, f, f_prime = mdl.u, mdl.u_prime, mdl.f, mdl.f_prime

# grid parameteres
grid_size = 200
grid = np.linspace(1e-5, 4, grid_size)


# shocks
shock_size = 250
shocks = np.exp(mu + s * np.random.randn(shock_size))


#init
np.random.seed(123)
g = np.random.uniform(low=1e-16, high=100, size=(grid_size,))

#tolerance and number of iterations
tol = 1e-16
iter = 2000

error = tol + 1
i = 0
errors = []

while error > tol and i < iter:
    g_1 = eul_updater(g, grid, u_prime, beta, f, f_prime, shocks)
    error = np.max(np.abs(g_1 - g))
    g = g_1
    print('iter ' + str(i) + '\n' + 'error: ' + str(error))
    errors.append(error)
    i += 1
    

#getting the value function
 

# ==plotting== #
    
#error
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(np.linspace(0, len(errors), num = len(errors)), errors, label="Error")
ax.legend()
ax.set_title('Error Rate over Iteration')
fig.savefig('./error.png')


#policy
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(grid, g, marker='o', label="Numerical Solution")
ax.plot(grid, mdl.c_star(grid), label="Analytical Solution")
ax.legend()
ax.set_title("Bellman Value Function: Numerical Solution vs Analytical Solution")
fig.savefig('./policy_comparison.png')


