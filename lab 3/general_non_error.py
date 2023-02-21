import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import pandas as pd
from sympy import symbols, diff
import sympy

dt = 0.01
DT = 1 / dt
x0 = 0.1
TLOW = 0
THIGH = 1

X, T = symbols('X T')
f = -2*X + sympy.exp(-2*T)


df = diff(f, X)
dft= diff(f,T)
dff = diff(df, X)

def returns_dydt(x, t):
    ############################################################################################################################
    dxdt = (1) * (-2*x + math.exp(-2*t))
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
            ans.append(ans[i - 1] + dt * f.evalf(subs={X: ans[i - 1], T:t[i-1]}))
        if order == 2:
            # second order
            ans.append(ans[i - 1] + dt * f.evalf(subs={X: ans[i - 1], T:t[i-1]}) 
                       + 0.5 * dt*dt* ((dft.evalf(subs={X: ans[i - 1], T:t[i-1]}) + (f.evalf(subs={X: ans[i-1],T:t[i-1]}))*df.evalf(subs={X: ans[i-1],T:t[i-1]}) )))
    #     if order == 3:
    #         # second order
    #         ans.append(ans[i - 1] + dt * f.evalf(subs={X: ans[i - 1],T:[i]})*((1 + 0.5 * dt * df.evalf(subs={X: ans[i - 1],T=t[i]})))
    #         + (1/6) * dt*dt*(df.evalf(subs={X: ans[i - 1],T:[i]})**2 + f.evalf(subs={X: ans[i - 1],T=t[i]}) * dff.evalf(subs={X: ans[i - 1],T:t[i]}))))
    return ans


def erer(ans):
    errr = [0]
    for i in range(1, int(DT)):
        errr.append((ans[i] - float(y1[i])))
    return errr


ans = euler(1)
errr = erer(ans)
npans = np.array(ans)
nperrr = np.array(errr)
# print(ans, errr)


ans2 = euler(2)
errr2 = erer(ans2)
#errr2= np.multiply(errr2, -1)

# ans3 = euler(3)
# errr3 = erer(ans3, y1)

print(ans)
print(errr)
print(ans2)
print(errr2)
# print(ans3)
# print(errr3)

# df = pd.DataFrame({"ans" : ans,
#                    "errr" : errr,
#                    "ans2" : ans2,
#                    "errr2": errr2,
#                    "ans3": ans3,
#                    "errr3": errr3,
#                    "Analytic": npy1})
# df.to_csv("lab_3.csv", index=False)

x = np.linspace(TLOW, THIGH, int(DT))

fig = plt.figure(figsize=(10, 10))


plt.plot(x, errr, color='green', label='1st', linewidth=2)
plt.plot(x, errr2, color='red', label='2nd', linewidth=2)
#plt.plot(x, errr3, color='blue', label='3rd', linewidth=2)

plt.xlabel('t')
plt.ylabel('Difference (Eulerr sol - Analytical sol)')
plt.legend()
plt.grid()
#plt.savefig('Lab3333')
plt.savefig('anq5.png')
plt.show()