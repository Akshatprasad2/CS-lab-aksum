import numpy as np
import matplotlib.pyplot as plt
import math

dt=0.01
DT=(int)(1/dt)

t= np.linspace(0,1,DT)
print(t)

x= np.exp(t)

print(x)

x1= [1]

for i in range (1,DT):
    x1.append(x1[i-1]+x1[i-1]*(t[i-1])*dt)


x2=[1]

for i in range (1,DT):
    x2.append(x2[i-1]+x2[i-1]*(-t[i])*dt + 0.5*x2[i-1]*(-t[i])*dt*(-1)*dt)


arrx1=np.array(x1)
print(arrx1)

arrx2=np.array(x2)
print(arrx2)

fig = plt.figure(figsize = (10, 10))

plt.plot(t,x) 
plt.plot(t,arrx1)
plt.plot(t,arrx2)
plt.xlabel("Time")
plt.ylabel("X's value")
plt.legend(["Exact","Euler","Euler2"])
plt.show()
