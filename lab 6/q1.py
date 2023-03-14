#write a code that create graph of a given function and points its maximum and minimum point of inflection and its value

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math


dt = 0.01

x0 = 0



TLOW = -5
THIGH = 5
DT = int((1 / dt) *(THIGH - TLOW))


x = np.linspace(TLOW, THIGH, DT)

#analitical soln
y1 = odeint(returns_dydt, x0, x)







