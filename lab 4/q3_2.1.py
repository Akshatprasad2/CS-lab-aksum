import numpy as np
import matplotlib.pyplot as plt
import math

# Define the constants and initial conditions
lambda_val = (math.log(2)/(22))
x0 = 1
t0 = 0
tf = 100
dt = 5

# Define the analytical solution
def analytical(t):
    return x0*np.exp(-lambda_val*t)

# Define the Euler method
def euler():
    x = np.zeros((int((tf-t0)/dt)+1,))
    t = np.zeros((int((tf-t0)/dt)+1,))
    x[0] = x0
    t[0] = t0
    for i in range(len(t)-1):
        x[i+1] = x[i] - lambda_val*x[i]*dt
        t[i+1] = t[i] + dt
    return x, t

# Define the second-order Taylor method
def taylor2():
    x = np.zeros((int((tf-t0)/dt)+1,))
    t = np.zeros((int((tf-t0)/dt)+1,))
    x[0] = x0
    t[0] = t0
    for i in range(len(t)-1):
        x[i+1] = x[i] - lambda_val*x[i]dt + 0.5*lambda_val2*x[i]*dt*2
        t[i+1] = t[i] + dt
    return x, t

# Define the third-order Taylor method
def taylor3():
    x = np.zeros((int((tf-t0)/dt)+1,))
    t = np.zeros((int((tf-t0)/dt)+1,))
    x[0] = x0
    t[0] = t0
    for i in range(len(t)-1):
        x[i+1] = x[i] - lambda_val*x[i]dt + 0.5*lambda_val2*x[i]*dt2 - (1/6)*lambda_val3*x[i]*dt*3
        t[i+1] = t[i] + dt
    return x, t

# Plot the results
t = np.linspace(t0, tf, int((tf-t0)/dt)+1)
x_analytical = analytical(t)
x_euler, _ = euler()
x_taylor2, _ = taylor2()
x_taylor3, _ = taylor3()

plt.plot(t, x_analytical, label="Analytical")
plt.plot(t, x_euler, label="Euler")
plt.plot(t, x_taylor2, label="Second-order Taylor")
plt.plot(t, x_taylor3, label="Third-order Taylor")
plt.xlabel("t")
plt.ylabel("x")
plt.legend()
plt.show()