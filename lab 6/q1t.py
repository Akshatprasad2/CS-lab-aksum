# take a function of x and plot the functio , its minima and maximas and its inflection points


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math



dt = 0.01

x0 = 0



TLOW = -5
THIGH = 5
DT = int((1 / dt) *(THIGH - TLOW))


