# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import fminbound


def bellman_updater(w, grid, u, beta, f, shocks):
    '''
    w : Our guess/Updated guess
    x_interp : The x_values we want in our interpolation
    u: utility (function)
    beta : discount factor (scalar)
    f : production function
    epsilon : shocks to production
    Tw : transformation of w

    Goal: Create a function which updates w -> Tw.
    Idea is that the sequence {w, Tw, T^2w,...} converges to the optimal value function
    '''
    # == initialize empty vectors == #
    # Value function mapping
    Tw = np.empty_like(w)

    #policy function
    pol = np.empty_like(w)

    # define a function to carry out linear interpolation
    def w_interp(x): return np.interp(x, grid, w)

    # minimize objective
    # Idea is that a consumer can atleast consume 0 units and at most consume y units.
    for i, y in enumerate(grid):

        # define our objective
        def obj(c):
            return - u(c) - beta * np.mean(w_interp(f(y - c) * shocks))

        # fminbound will optimize and get out c_star
        # we need to minimize the negative of the objective function to maximize the objective
        c_star = fminbound(obj, 1e-10, y)

        # get our Tw
        Tw[i] = -obj(c_star)

        pol[i] = c_star

    return Tw, pol
