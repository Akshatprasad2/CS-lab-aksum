import numpy as np
import matplotlib.pyplot as plt

def euler_method(xA_initial, xB_initial, tau_A, tau_B, delta_t, t_end):
    t = np.arange(0, t_end, delta_t)
    xA = np.zeros(len(t))
    xB = np.zeros(len(t))
    xA[0] = xA_initial
    xB[0] = xB_initial

    for i in range(1, len(t)):
        xA_dot = -xA[i-1] / tau_A
        xB_dot = -xA_dot - (xB[i-1] / tau_B)
        xA[i] = xA[i-1] + xA_dot * delta_t
        xB[i] = xB[i-1] + xB_dot * delta_t

    return t, xA, xB

# Case 1: tau_A = tau_B
tau_A1 = 1
tau_B1 = 1
delta_t1 = 0.01
t_end1 = 10

# Case 2: tau_A ≃ 10^4 * tau_B
tau_A2 = 10**4
tau_B2 = 1
delta_t2 = 0.01
t_end2 = 150

# Case 3: tau_A ≃ 10^(-4) * tau_B
tau_A3 = 10**(-4)
tau_B3 = 1
delta_t3 = 0.0001
t_end3 = 0.5

# Perform Euler's method integration for each case
t1, xA1, xB1 = euler_method(1, 0, tau_A1, tau_B1, delta_t1, t_end1)
t2, xA2, xB2 = euler_method(1, 0, tau_A2, tau_B2, delta_t2, t_end2)
t3, xA3, xB3 = euler_method(1, 0, tau_A3, tau_B3, delta_t3, t_end3)

# Plot xA(t) and xB(t) for each case
plt.figure(figsize=(15, 5))
cases = [(t1, xA1, xB1), (t2, xA2, xB2), (t3, xA3, xB3)]
titles = ["tau_A = tau_B", "tau_A ≃ 10^4 * tau_B", "tau_A ≃ 10^(-4) * tau_B"]

for i, (t, xA, xB) in enumerate(cases):
    plt.subplot(1, 3, i+1)
    plt.plot(t, xA, label="xA(t)")
    plt.plot(t, xB, label="xB(t)")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(titles[i])
    plt.legend()

plt.tight_layout()
plt.show()

# Plot xA-xB graph for each case
plt.figure(figsize=(15, 5))

for i, (t, xA, xB) in enumerate(cases):
    plt.subplot(1, 3, i+1)
    plt.plot(xA, xB)
    plt.xlabel("xA")
    plt.ylabel("xB")
    plt.title(titles[i])

plt.tight_layout()
plt.show()