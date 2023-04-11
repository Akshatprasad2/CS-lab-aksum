import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

from sympy import symbols, diff
import sympy


dt = 0.01


x0 = 0
v0 = 1

fig = plt.figure(figsize=(10, 10))

TLOW = 0
THIGH = 10
DT = int((1 / dt) *(THIGH - TLOW))

# k = omega^2
T=1
om = 2*math.pi/T

k = om*om
b = k


def X_dot(x,v,b):
    return v

def V_dot(x,v,b):
    return -2*b*v - k*x


t = np.linspace(TLOW, THIGH, DT)



def euler(t, dt, order,b):
    ans_x = [x0]
    ans_v = [v0]
    et= [(v0**2/2) + k*x0**2/2]

    for i in range(1, DT):
        if order == 1:
            # first order
            ans_v.append(ans_v[i-1] + dt*V_dot(ans_x[i-1],ans_v[i-1],b))
            ans_x.append(ans_x[i - 1] + dt * X_dot(ans_x[i-1],ans_v[i-1],b))
            et.append((ans_v[i]**2/2) + k*ans_x[i]**2/2)
        
    return ans_x, ans_v, et






b_array= [20, 7, 2*math.pi, 5.8 ,1]
colorr= ['steelblue', 'khaki','springgreen','violet','orange']

for b in range(len(b_array)):
    ans_x, ans_v,et = euler(t, dt, 1,b_array[b])  
    

    plt.plot(t, ans_x, color=colorr[b], label=(f"b= {b_array[b]}"), linewidth=2)
    #plt.plot(t, ans_v, color=colorr[b], label=(f"b= {b_array[b]}"), linewidth=2)
    #plt.plot(ans_x, ans_v, color=colorr[b], label=(f"b= {b_array[b]}"), linewidth=2)
    #plt.plot(t, et, color=colorr[b], label=(f"b= {b_array[b]}"), linewidth=2)

# print(ansx)
# print(ansy)

t1 = np.linspace(TLOW, THIGH, DT)


#plotting


#plt.plot(t, ansx, color='blue', label='b= omega', linewidth=2)


#plt.plot(t, ansx, color='red', label='b= omega', linewidth=2)



#plt.plot(t, ansv, color='red', label='x(b)', linewidth=2)

#plt.plot(ansx,ansy, color='green', label='x(a) vs x(b)', linewidth=2)



plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.grid()
plt.savefig('q2_a.png')
plt.show()