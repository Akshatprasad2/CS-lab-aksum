import numpy as np
import matplotlib.pyplot as plt

def euler_method(xA_initial, xB_initial, tau, delta_t, t_end):
    t = np.arange(0, t_end, delta_t)
    xA = np.zeros(len(t))
    xB = np.zeros(len(t))
    xA[0] = xA_initial
    xB[0] = xB_initial

    for i in range(1, len(t)):
        xA_dot = (xB[i-1] - xA[i-1]) / tau
        xB_dot = -xA_dot
        xA[i] = xA[i-1] + xA_dot * delta_t
        xB[i] = xB[i-1] + xB_dot * delta_t

    return t, xA, xB

# Parameters
tau = 1
delta_t = 0.01
t_end = 10

# Perform Euler's method integration
t, xA, xB = euler_method(100, 0, tau, delta_t, t_end)

# Plot xA(t) and xB(t) separately
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(t, xA)
plt.xlabel("t")
plt.ylabel("xA(t)")
plt.title("xA(t) vs t")

plt.subplot(1, 2, 2)
plt.plot(t, xB)
plt.xlabel("t")
plt.ylabel("xB(t)")
plt.title("xB(t) vs t")

plt.tight_layout()
plt.show()

# Plot xA(t) and xB(t) together
plt.figure()
plt.plot(t, xA, label="xA(t)")
plt.plot(t, xB, label="xB(t)")
plt.xlabel("t")
plt.ylabel("x")
plt.title("xA(t) and xB(t) vs t")
plt.legend()
plt.show()

# Plot xA-xB graph
plt.figure()
plt.plot(xA, xB)
plt.xlabel("xA")
plt.ylabel("xB")
plt.title("xA vs xB")
plt.show()