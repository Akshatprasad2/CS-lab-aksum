import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import pandas as pd
from sympy import symbols, diff
import sympy


dt = 10**

x0 = 1

TLOW = 0
THIGH = 10

DT = int((1 / dt) *(THIGH - TLOW))

#here f is the dx/dt function
X, T = symbols('X T')
f = 40

def returns_dydt(x, t):
    ############################################################################################################################
    dxdt = 40
    return dxdt

df = diff(f, X)
dft= diff(f,T)
dfx = diff(df, X)

t = np.linspace(TLOW, THIGH, DT +1)

#analitical soln
y1 = odeint(returns_dydt, x0, t)


def tailor(order):
    ans = [x0]
    for i in range(1, DT):
        if order == 1:
            # first order
            ans.append(ans[i - 1] + dt * returns_dydt(ans[i-1],t[i-1]))
        if order == 2:
            # second order
            ans.append(ans[i - 1] + dt * returns_dydt(ans[i-1],t[i-1]) 
                       + 0.5 * dt*dt* ((dft.evalf(subs={X: ans[i - 1], T:t[i-1]}) + (returns_dydt(ans[i-1],t[i-1]))*df.evalf(subs={X: ans[i-1],T:t[i-1]}) )))
        
    return ans


def erer(ans):
    errr = [0]
    for i in range(1, DT):
        errr.append(ans[i] - float(y1[i]))
    return errr

ans = tailor(1)
errr= erer(ans)

print(ans)
print(errr)

#note this is for x span of 1 sec 
#for x span of 10 sec

t1 = np.linspace(TLOW, THIGH, DT)
#plotting
fig = plt.figure(figsize=(10, 10))

plt.plot(t1, ans, color='green', label='1st', linewidth=2)

plt.xlabel('t')
plt.ylabel('Analytical sol')
plt.legend()
plt.grid()
plt.savefig('q3.png')
plt.show()
