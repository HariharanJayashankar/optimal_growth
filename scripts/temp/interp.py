# -*- coding: utf-8 -*-
"""
Testing Interp

@author: Hariharan
"""


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)

plt.plot(x, y)

xval = np.linspace(0, 2*np.pi, 5)
y_interp = np.interp(xval, x, y)

plt.plot(x, y, 'o')
plt.plot(xval, y_interp, '-')

plt.show()