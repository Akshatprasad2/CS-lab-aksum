import numpy as np
import matplotlib.pyplot as plt

from sympy import symbols, diff
import sympy


dt = 0.01

x0 = 0

TLOW = -2
THIGH = 2

DT = int((1 / dt) *(THIGH - TLOW))

X, T = symbols('X T')
f = X**2 - 1

def f(x):
    return x**2 - 1   # Example function

  # x-axis values

x = np.linspace(TLOW, THIGH, DT+1)
y = f(x)                # y-axis values

# Plot the function

fig = plt.figure(figsize=(10, 10))

ax = plt.plot()

plt.plot(x, [0 for i in range(len(x))], color='black', linewidth=2)
plt.plot(x, y, color='blue', linewidth=2)

# Mark point of infection
root1 = -1   # Example point of infection
root2 = 1   # Example point of infection

plt.scatter(root1, f(root1), color='green', s=100, label = 'Stable Point')
plt.scatter(root2, f(root2), color='red', s=100, label = 'Unstable Point')

# Set axis labels and title
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function Plot with Infection Point')
plt.grid()
plt.legend()
plt.show()
