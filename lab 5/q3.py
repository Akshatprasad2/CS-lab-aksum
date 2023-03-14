import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

from sympy import symbols, diff
import sympy


dt = 5

x0 = 3*10**(9)

TLOW = 0
THIGH = 600

A= 0.03
B= 3*10**(-12)

DT = int((1 / dt) *(THIGH - TLOW))



#here f is the dx/dt function
X, T = symbols('X T')
f = A*X -B*X*X

def returns_dydt(x, t):
    ############################################################################################################################
    dxdt = A*x - B*x*x
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


def erer(ans):
    errr = [0]
    for i in range(1, DT):
        errr.append(ans[i] - float(y1[i]))
    return errr

ans1 = tailor2(1)
ans2 = tailor2(2)
ans3 = tailor2(3)

err1 = erer(ans1)
err2 = erer(ans2)
err3 = erer(ans3)

print(ans3)
print()
print(y1)

print(len(ans3))
print(len(y1))


t1 = np.linspace(TLOW, THIGH, DT)


#plotting
fig = plt.figure(figsize=(10, 10))

plt.plot(t1, ans1, color='blue', label='1st', linewidth=2)
plt.plot(t1, ans2, color='red', label='2st', linewidth=2)
plt.plot(t1, ans3, color='green', label='3st', linewidth=2)
plt.plot(t1, y1[:len(y1)-1], color='orange', label='analytical', linewidth=2)


# plt.plot(t1, err1, color='red', label='1st', linewidth=2)
# plt.plot(t1, err2, color='blue', label='2st', linewidth=2)
# plt.plot(t1, err3, color='green', label='3st', linewidth=2)



plt.xlabel('t(year)')
#plt.ylabel('Analytical sol(amount)')

plt.ylabel('Analytical sol')
plt.legend()
plt.grid()
#plt.savefig('q3.png')
plt.show()