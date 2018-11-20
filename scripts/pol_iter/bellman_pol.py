# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import brentq

def eul_updater(g,
                grid,
                u_prime,
                beta,
                f,
                f_prime,
                shocks):
    
    '''
    Updates the euler function in a dynamic optimization setting
    
    g: inputs of initial guesses for the policy function
    grid: grid points for the policy function
    u: utility function
    u_prime: derivative of utility wrt c
    beta: discount rate
    f: production function
    f_prime: derivative of production function wrt c
    shocks: exogenous shocks
    '''
    
    Kg = np.empty_like(g)
    
    def g_interp(x): return np.interp(x, grid, g)
    
    for i, y in enumerate(grid):
        
        def obj(c):
            rhs = u_prime(g_interp(f(y - c) * shocks)) * f_prime(y - c)*shocks
            return u_prime(c) - beta * np.mean(rhs)
            
        c_star = brentq(obj, 1e-10, y)
        
        Kg[i] = c_star
        
        
    return Kg
    