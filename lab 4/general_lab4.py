import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

dt = 0.01
DT = 1 / dt
X_INITIAL = 0
TIME_LOWER_LIM = 0
TIME_UPPER_LIM = 10
TIME_DIFF = TIME_UPPER_LIM - TIME_LOWER_LIM

# linear differ. eq integrator
def calc_LDE(x, t):
    doit = 40
    return doit

#function returns fn value
def fn(x):
    return 40


t = np.linspace(TIME_LOWER_LIM, TIME_UPPER_LIM, int(DT) + 1)
#call integrator & returns its value list
y1 = odeint(calc_LDE, X_INITIAL, t)


#calculate euler solution
def euler(dt):
    DT=1/dt
    ans = [X_INITIAL]
    for i in range(1, int(DT) * TIME_DIFF):
        ans.append(ans[i - 1] + dt * fn(ans[i - 1]))
    return ans

#calculate error
def calculate_error(analytical, y):
    errr = [0]
    for i in range(1, int(DT)):
        errr.append((analytical[i] - float(y[i])))
    return errr


ans = euler()
errr = calculate_error(ans, y1)
x = np.linspace(TIME_LOWER_LIM, TIME_UPPER_LIM, int(DT) * TIME_DIFF)
fig = plt.figure(figsize=(10, 10))
plt.plot(x, ans, color='green', label='Δt = 0.1', linewidth=2)
plt.xlabel('t')
plt.ylabel('Difference (Eulerr sol - Analytical sol)')
plt.legend()
plt.grid()
plt.savefig('lab4_1e.png')
plt.show()