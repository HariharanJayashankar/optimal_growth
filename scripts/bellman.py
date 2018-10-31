# -*- coding: utf-8 -*-
"""
Bellman Updater

@author: Hariharan
"""
import numpy as np
from scipy.optimize import fminbound

def bellman_updater(y, u, beta, f, epsilon, Tw):
    '''
    w : Our guess/Updated guess
    y : income (vector)
    u: utility (function)
    beta : discount factor (scalar)
    f : production function
    epsilon : shocks to production
    Tw : transformation of w

    Goal: Create a function which updates w -> Tw. 
    Idea is that the sequence {w, Tw, T^2w,...} converges to the optimal value function
    '''
 	
 	#define objective function
    obj = lambda c: u(c) + beta * np.mean(f(y - c)*epsilon)

    #minimize objective
    #Idea is that a consumer can atleast consume 0 units and at most consume y units.
    c_star = fminbound(-obj, 0, y)
    Tw = obj(c_star)

    return Tw, c_star

