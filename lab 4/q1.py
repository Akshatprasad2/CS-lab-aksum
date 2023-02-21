import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import pandas as pd
from sympy import symbols, diff
import sympy


dt = 0.01
DT = (1 / dt) 
x0 = 1

TLOW = 0
THIGH = 1

#here f is the dx/dt function
X, T = symbols('X T')
f = T*T - X

def returns_dydt(x, t):
    ############################################################################################################################
    dxdt = t*t - x
    return dxdt

df = diff(f, X)
dft= diff(f,T)
dfx = diff(df, X)

t = np.linspace(TLOW, THIGH, int(DT) +1)

#analitical soln
y1 = odeint(returns_dydt, x0, t)


def tailor(order):
    ans = [x0]
    for i in range(1, int(DT)):
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
    for i in range(1, int(DT)):
        errr.append(ans[i] - float(y1[i]))
    return errr

ans = tailor(1)
errr= erer(ans)

print(ans)
print(errr)

t1 = np.linspace(TLOW, THIGH, int(DT))
#plotting
fig = plt.figure(figsize=(10, 10))

plt.plot(t1, errr, color='green', label='1st', linewidth=2)
plt.xlabel('t')
plt.ylabel('Analytical sol')
plt.legend()
plt.grid()
plt.savefig('1st.png')
plt.show()
