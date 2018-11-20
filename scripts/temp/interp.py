# -*- coding: utf-8 -*-
"""
Testing Interp

@author: Hariharan

x, y can imagined to be the actual data we have from which we want to
approximate a function. 

xval is the set of xvalues at which we approximate the function.

len(xval) > x
"""


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 10)
y = np.sin(x)

xval = np.linspace(0, 2*np.pi, 50)
y_interp = np.interp(xval, x, y)

plt.plot(x, y, 'o')
plt.plot(xval, y_interp, '-')

plt.show()