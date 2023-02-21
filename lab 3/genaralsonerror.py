import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import pandas as pd
from sympy import symbols, diff

dt = 0.1
DT = 1 / dt
x0 = 1
TLOW = 0
THIGH = 1

X = symbols('X')
    ############################################################################################################################
f = -X
df = diff(f, X)
dff = diff(df, X)

def returns_dydt(x, t):
    ############################################################################################################################
    dxdt = (1) * (-x)
    return dxdt


t = np.linspace(TLOW, THIGH, int(DT) + 1)
y1 = odeint(returns_dydt, x0, t)
npy1 = np.array(y1[:-1])
npy1 = npy1.flatten()


def euler(order):
    ans = [x0]
    for i in range(1, int(DT)):
        if order == 1:
            # first order
            ans.append(ans[i - 1] + dt * f.evalf(subs={X: ans[i - 1]}))
        if order == 2:
            # second order
            ans.append(ans[i - 1] + dt * f.evalf(subs={X: ans[i - 1]})*(1 + 0.5 * dt * df.evalf(subs={X: ans[i - 1]})))
        if order == 3:
            # second order
            ans.append(ans[i - 1] + dt * f.evalf(subs={X: ans[i - 1]})*((1 + 0.5 * dt * df.evalf(subs={X: ans[i - 1]}))
            + (1/6) * dt*dt*(df.evalf(subs={X: ans[i - 1]})**2 + f.evalf(subs={X: ans[i - 1]}) * dff.evalf(subs={X: ans[i - 1]}))))
    return ans


def erer(ans, y):
    errr = [0]
    for i in range(1, int(DT)):
        errr.append((ans[i] - float(y[i])))
    return errr


ans = euler(1)
errr = erer(ans, y1)
npans = np.array(ans)
nperrr = np.array(errr)
# print(ans, errr)
x = np.linspace(TLOW, THIGH, int(DT))

ans2 = euler(2)
errr2 = erer(ans2, y1)

ans3 = euler(3)
errr3 = erer(ans3, y1)

print(ans)
print(errr)
print(ans2)
print(errr2)
print(ans3)
print(errr3)

df = pd.DataFrame({"ans" : ans,
                   "errr" : errr,
                   "ans2" : ans2,
                   "errr2": errr2,
                   "ans3": ans3,
                   "errr3": errr3,
                   "Analytic": npy1})
df.to_csv("lab_3.csv", index=False)

fig = plt.figure(figsize=(10, 10))

plt.plot(x, errr, color='green', label='1st', linewidth=2)
plt.plot(x, errr2, color='red', label='2nd', linewidth=2)
plt.plot(x, errr3, color='blue', label='3rd', linewidth=2)

plt.xlabel('t')
plt.ylabel('Difference (Eulerr sol - Analytical sol)')
plt.legend()
plt.grid()
plt.savefig('Lab3333')
plt.show()