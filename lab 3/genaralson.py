

from sympy import symbols, diff
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

dt = 0.01
DT = 1 / dt
TLOW = 0
THIGH = 1
x_init = 1

X = symbols('X')
    ############################################################################################################################
f = -X
df = diff(f, X)
dff = diff(df, X)


def returns_dydt(x, t):
    ############################################################################################################################
    dxdt = -x
    return dxdt


t = np.linspace(TLOW, THIGH, int(DT) + 1)
y = odeint(returns_dydt, x_init, t)
npy1 = np.array(y[:-1])
npy1 = npy1.flatten()


def euler(order):
    ans = [x_init]
    for i in range(1, int(DT)):
        if order == 1:
            # first order
            ans.append(ans[i - 1] + dt * f.evalf(subs={X: ans[i - 1]}))
        if order == 2:
            # second order
            ans.append(ans[i - 1] + dt * f.evalf(subs={X: ans[i - 1]})*(1 + 0.5 * dt * df.evalf(subs={X: ans[i - 1]})))
        if order == 3:
            # second order
            ans.append(ans[i - 1] + dt * f.evalf(subs={X: ans[i - 1]})*(1 + 0.5 * dt * df.evalf(subs={X: ans[i - 1]}))
            + (1/6) * dt*dt*(df.evalf(subs={X: ans[i - 1]})**2 + f.evalf(subs={X: ans[i - 1]}) * dff.evalf(subs={X: ans[i - 1]})))
    return ans


ans1 = euler(1)
ans2 = euler(2)
ans3 = euler(3)
x = np.linspace(TLOW, THIGH, int(DT))

fig = plt.figure(figsize=(10, 10))
plt.plot(x, ans1, color='blue', label="Eular's Method for Δt = 0.01 1st Order", linewidth=2)
plt.plot(x, ans2, color='red', label="Eular's Method for Δt = 0.01 2nd Order", linewidth=2)
plt.plot(x, ans3, color='black', label="Eular's Method for Δt = 0.01 3nd Order", linewidth=2)
plt.plot(x, npy1, color='green', label='Analytical Solution', linewidth=2)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.grid()
plt.savefig('Q52 - Anlyt')
plt.show()