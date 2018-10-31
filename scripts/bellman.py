# -*- coding: utf-8 -*-
"""
Bellman Updater

@author: Hariharan
"""

from scipy.optimize import fminbound

def bellman_updater(w, y, u, beta, f, epsilon, Tw):
    '''
    w : Our guess/Updated guess
    y : income (vector)
    u: utility (function)
    beta: discount factor (scalar)
    '''