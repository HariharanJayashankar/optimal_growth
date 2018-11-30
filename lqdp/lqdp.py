import numpy as np
from quantecon import LQ
import matplotlib.pyplot as plt
import os

# == specifying paths/wd == #
# this specifies the working directory as the script directory
script_path = os.path.abspath(__file__)
os.chdir(os.path.split(script_path)[0])

# == Model Parameters ==  #

r		= 0.05
beta 	= 1 / (1 + r)
c_bar 	= 2
mu 		= 1
sigma 	= 0.25
T 		= 45
q 		= 1e6

# == Initializing the model == #

A 	= np.array([[1 + r, -c_bar + mu],
              [0, 1]])

B 	= np.array([[-1],
              [0]])

C 	= np.array([[sigma],
              [0]])

R 	= np.zeros((2, 2))

Rf 	= np.array([[q, 0],
               [0, 0]])

Q = 1



# == Solving the model = #

lq 	= LQ(Q, R, A, B, C, beta = beta, T = T, Rf = Rf)
x0 	= (0, 1)
xp, up, wp 	= lq.compute_sequence(x0)

a = xp[0, ]
c = up.flatten() + c_bar
inc = sigma * wp[0, 1:] + mu

# == plotting == #

fig, ax = plt.subplots(figsize = (9, 6))
ax.plot(list(range(1, T+1)), inc, 'g-', label="income")
ax.plot(list(range(T)), c, 'k-', label="consumption")
ax.legend()
plt.savefig('./incomeVsCons.png')

fig, ax = plt.subplots(figsize = (9, 6))
ax.plot(list(range(1, T+1)), inc, 'g-', label="income")
ax.plot(list(range(T+1)), a, 'k-', label="assets")
ax.legend()
plt.savefig('./incVsAssets.png')

fig, ax = plt.subplots(figsize = (9, 6))
ax.plot(list(range(1, T+1)), np.cumsum(inc - mu), 'g-', label="Cumulative Income Shock (de-meaned)")
ax.plot(list(range(T+1)), a, 'k-', label="Assets")
ax.legend()
plt.savefig('./cumIncVsAssets.png')
