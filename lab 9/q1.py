import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

from sympy import symbols, diff
import sympy


dt = 0.1


x0 = 1
v0 = 0

fig = plt.figure(figsize=(10, 10))

TLOW = 0
THIGH = 100
DT = int((1 / dt) *(THIGH - TLOW))

# k = omega^2
T=10
om = 2*math.pi/T
k = om*om




def X_dot(x,v,k):
    return v

def V_dot(x,v,k):
    return - k*x


t = np.linspace(TLOW, THIGH, DT)



def euler(t, dt, order,k):
    ans_x = [x0]
    ans_v = [v0]
    et= [(v0**2/2) + k*x0**2/2]
    

    for i in range(1, DT):
        if order == 1:
            # first order
            ans_v.append(ans_v[i-1] + dt*V_dot(ans_x[i-1],ans_v[i-1],k))
            ans_x.append(ans_x[i - 1] + dt * X_dot(ans_x[i-1],ans_v[i-1],k))
            et.append((ans_v[i]**2/2) + k*ans_x[i]**2/2)
        
    return ans_x, ans_v, et



# for b in range(len(b_array)):
#     ans_x, ans_v,et = euler(t, dt, 1,b_array[b])  
    

#     plt.plot(t, ans_x, color=colorr[b], label=(f"b= {b_array[b]}"), linewidth=2)
#     #plt.plot(t, ans_v, color=colorr[b], label=(f"b= {b_array[b]}"), linewidth=2)
#     #plt.plot(ans_x, ans_v, color=colorr[b], label=(f"b= {b_array[b]}"), linewidth=2)
#     #plt.plot(t, et, color=colorr[b], label=(f"b= {b_array[b]}"), linewidth=2)

ans_x,ans_v,et = euler(t, dt, 1,k)

loget= np.log(et)

# plt.plot(t, ans_x, color='blue', label=(f"X"), linewidth=2)
# plt.plot(t, ans_v, color='red', label=(f"V"), linewidth=2)
# plt.plot(ans_x, ans_v, label=(f"T= {T}"), linewidth=2)
# plt.plot(t, et, color='green', label=(f"T= {T}"), linewidth=2)
plt.plot(t, loget, color='green', label=(f"T= {T}"), linewidth=2)

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
plt.savefig('q1cc_4.png')
plt.show()