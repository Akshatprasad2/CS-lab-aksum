import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import pandas as pd
from sympy import symbols, diff
import sympy


dt = 10**9

x0 = 1

TLOW = 0
THIGH = 20*10**(9)

DT = int((1 / dt) *(THIGH - TLOW))



#here f is the dx/dt function
X, T = symbols('X T')
f = -(sympy.ln(2)/(4.5*10**9))*X

def returns_dydt(x, t):
    ############################################################################################################################
    dxdt = -(math.log(2)/(4.5*10**9))*x
    return dxdt

df = diff(f, X)
dft= diff(f,T)
dfx = diff(df, X)

t = np.linspace(TLOW, THIGH, DT +1)

#analitical soln
y1 = odeint(returns_dydt, x0, t)


def tailor(order):
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
            + (1/6) * dt*dt*(df.evalf(subs={X: ans[i - 1]})**2 + f.evalf(subs={X: ans[i - 1]}) * dfx.evalf(subs={X: ans[i - 1]}))))
    return ans
                        

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


def erer(ans):
    errr = [0]
    for i in range(1, DT):
        errr.append(ans[i] - float(y1[i]))
    return errr

ans2 = tailor2(2)
ans1 = tailor2(1)
ans3 = tailor2(3)


errr1= erer(ans1)
errr2= erer(ans2)
errr3= erer(ans3)



# print(ans)
# print(errr)


t1 = np.linspace(TLOW, THIGH, DT)
xt= np.exp((-np.log(2)/(1600))*t1)

print(xt)
print(ans1)
print(ans2)
#print(ans3)
#plotting
fig = plt.figure(figsize=(10, 10))

# plt.plot(t1, ans1, color='yellow', label='1st', linewidth=2)
# plt.plot(t1, ans2, color='blue', label='2st', linewidth=2)
# plt.plot(t1, ans3, color='green', label='3st', linewidth=2)
# plt.plot(t, y1, color='red', label='real',linewidth=2)
# plt.plot(t1, xt, color='red', label='real',linewidth=2)

plt.plot(t1, errr1, color='yellow', label='1st', linewidth=2)
plt.plot(t1, errr2, color='blue', label='2st', linewidth=2)
plt.plot(t1, errr3, color='green', label='3st', linewidth=2)




plt.xlabel('t(year)')
#plt.ylabel('Analytical sol(amount)')
plt.ylabel('Analytical sol(error)')
plt.legend()
plt.grid()
plt.savefig('q3arror.png')
plt.show()
