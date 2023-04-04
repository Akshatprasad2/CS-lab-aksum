import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

from sympy import symbols, diff
import sympy


dt = 0.5

x0 = 0

TLOW = 0
THIGH = 25


DT = int((1 / dt) *(THIGH - TLOW))



#here f is the dx/dt function
g= 9.8
k= 0.18
m= 110

X, T = symbols('X T')
f = g - k*X*X/m 

def returns_dydt(x, t):
    ############################################################################################################################
    dxdt = g - k*x*x/m
    return dxdt

df = diff(f, X)
dft= diff(f,T)
dfx = diff(df, X)

t = np.linspace(TLOW, THIGH, DT+1)

#analitical soln
y1 = odeint(returns_dydt, x0, t)




def tailor2(order):
    ans = [x0]
    for i in range(1, DT):
        if order == 1:
            # first order
            ans.append(ans[i - 1] + dt * returns_dydt(ans[i-1],t[i-1]))
        if order == 2:
            # second order
            ans.append(ans[i - 1] + dt * returns_dydt(ans[i-1],t[i-1]) 
                       + 0.5 * dt*dt* ((dft.evalf(subs={X: ans[i - 1], T:t[i-1]}) + (returns_dydt(ans[i-1],t[i-1]))*df.evalf(subs={X: ans[i-1],T:t[i-1]}) )))
        
        if order == 3:
            #3rd order
            ans.append(ans[i - 1] + dt * returns_dydt(ans[i-1],t[i-1])
                          + 0.5 * dt*dt* ((dft.evalf(subs={X: ans[i - 1], T:t[i-1]}) + (returns_dydt(ans[i-1],t[i-1]))*df.evalf(subs={X: ans[i-1],T:t[i-1]}) ))
                          + (1/6) * dt*dt*dt*(returns_dydt(ans[i-1],t[i-1])*(df.evalf(subs={X: ans[i-1],T:t[i-1]})**2)  +  (returns_dydt(ans[i-1],t[i-1])**2)*(dfx.evalf(subs={X: ans[i-1],T:t[i-1]})) )
            )

    return ans