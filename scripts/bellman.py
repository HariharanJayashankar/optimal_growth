# -*- coding: utf-8 -*-
"""
Bellman Updater

@author: Hariharan
"""
import numpy as np
from scipy.optimize import fminbound


def bellman_updater(w, grid, u, beta, f, shocks, Tw = None):
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
    #initialize empty vectors
    if Tw is None:
        Tw = np.empty_like(w)

    pol = np.empty_like(w)

    #define a function to carry out linear interpolation
    w_interp = lambda x: np.interp(x, grid, w)

    #minimize objective
    #Idea is that a consumer can atleast consume 0 units and at most consume y units.
    for i, y in enumerate(grid):

        #define our objective
        def obj(c):
            return - u(c) - beta * np.mean(w_interp(f(y - c) * shocks))

        #fminbound will optimize and get out c_star
        c_star = fminbound(obj, 1e-10, y) #we need to minimize the negative of the objective function to maximize the objective

        #get our Tw
        Tw[i] = -obj(c_star)

        pol[i] = c_star

    return Tw, pol
