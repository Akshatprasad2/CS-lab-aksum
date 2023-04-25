import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

from sympy import symbols, diff
import sympy

width, height = 10, 10
t = np.linspace(0, 2*np.pi, 100)
plt.plot(width*np.sin(t), height*np.cos(t))
plt.grid()
plt.show()