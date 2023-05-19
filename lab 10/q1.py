import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

from sympy import symbols, diff
import sympy


dt = 60*60*25  #dt 1 hrs

m= 2*10**30
G= 6.67*10**(-11)
r= 1.5*10**11

x0 = r
y0 = 0

vx0=0
vy0=30000

fig = plt.figure(figsize=(10, 10))

TLOW = 0
THIGH = 1100*60*60*24
DT = int((1 / dt) *(THIGH - TLOW))






def X_dot(x,y,vx,vy):
    return vx

def Y_dot(x,y,vx,vy):
    return vy

def Vx_dot(x,y,vx,vy):
    return -G*m*x/(x**2+y**2)**(3/2)

def Vy_dot(x,y,vx,vy):
    return -G*m*y/(x**2+y**2)**(3/2)


t = np.linspace(TLOW, THIGH, DT)

def euler(t, dt, order,k):
    ans_x = [x0]
    ans_y =[y0]
    ans_vx = [vx0]
    ans_vy = [vy0]
    

    for i in range(1, DT):
        if order == 1:
            # first order
            ans_x.append(ans_x[i-1] + dt*X_dot(ans_x[i-1],ans_y[i-1],ans_vx[i-1],ans_vy[i-1]))
            ans_y.append(ans_y[i-1] + dt*Y_dot(ans_x[i-1],ans_y[i-1],ans_vx[i-1],ans_vy[i-1]))
            ans_vx.append(ans_vx[i-1] + dt*Vx_dot(ans_x[i-1],ans_y[i-1],ans_vx[i-1],ans_vy[i-1]))
            ans_vy.append(ans_vy[i-1] + dt*Vy_dot(ans_x[i-1],ans_y[i-1],ans_vx[i-1],ans_vy[i-1]))
            
        
    return ans_x, ans_y, ans_vx, ans_vy



ans_x, ans_y, ans_vx, ans_vy = euler(t, dt, 1,1)

plt.plot(t, ans_x, color='blue', label=(f"X"), linewidth=2)
plt.plot(t, ans_y, color='red', label=(f"Y"), linewidth=2)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.grid()
# plt.savefig('q1_1.png')
# plt.show()

fig = plt.figure(figsize=(10, 10))
plt.plot(t,ans_vx, color='green', label=(f"Vx"), linewidth=2)
plt.plot(t,ans_vy, color='yellow', label=(f"Vy"), linewidth=2)
plt.xlabel('t')
plt.ylabel('v(t)')
plt.legend()
plt.grid()
# plt.savefig('q1_2.png')
# plt.show()

fig = plt.figure(figsize=(15, 15))
plt.plot(ans_x,ans_y, color='red', label=("Orbit"), linewidth=1)
width, height = r, r
t = np.linspace(0, 2*np.pi, 100)
plt.plot(width*np.sin(t), height*np.cos(t), color='blue', label=("Analytical "), linewidth=2)
plt.plot(0, 0,marker="o", markersize=30, markeredgecolor="red", markerfacecolor="yellow",label=("Sun"))
plt.plot(r, 0,marker="o", markersize=15, markeredgecolor="red", markerfacecolor="blue",label=("earth"))
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.grid()
plt.savefig('q1_3C.png')
plt.show()