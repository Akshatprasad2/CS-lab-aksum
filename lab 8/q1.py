import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

from sympy import symbols, diff
import sympy


dt = 0.01

tou1 =1
tou2 =1

x0 = 100
y0 = 0


TLOW = 0
THIGH = 10
DT = int((1 / dt) *(THIGH - TLOW))




def X_dot(x,y):
    return -x/tou1

def Y_dot(x,y):
    return -X_dot(x,y) -(y)/tou2


t = np.linspace(TLOW, THIGH, DT)



def euler(t, dt, order):
    ans_x = [x0]
    ans_y = [y0]

    for i in range(1, DT):
        if order == 1:
            # first order
            ans_y.append(ans_y[i-1] + dt*Y_dot(ans_x[i-1],ans_y[i-1]))
            ans_x.append(ans_x[i - 1] + dt * X_dot(ans_x[i-1],ans_y[i-1]))
            
        
    return ans_x, ans_y




ansx, ansy = euler(t, dt, 1)

# print(ansx)
# print(ansy)

t1 = np.linspace(TLOW, THIGH, DT)


#plotting
fig = plt.figure(figsize=(10, 10))

plt.plot(t, ansx, color='blue', label='x(a)', linewidth=2)
plt.plot(t, ansy, color='red', label='x(b)', linewidth=2)

#plt.plot(ansx,ansy, color='green', label='x(a) vs x(b)', linewidth=2)



plt.xlabel('x(b)')
plt.ylabel('t')
plt.legend()
plt.grid()
#plt.savefig('q2b.png')
plt.show()