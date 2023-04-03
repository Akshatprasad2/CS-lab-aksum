import math
import matplotlib.pyplot as plt

# Constants
v0 = 700  # initial velocity (m/s)
Bm = 4e-5  # air drag coefficient (m^-1)
#Bm = 0  # air drag coefficient (m^-1)
g = 9.8  # acceleration due to gravity (m/s^2)
m = 1  # mass of projectile (kg)

fig = plt.figure(figsize=(10, 10))
# Euler method
def euler_step(vx, vy, dt):
    v = math.sqrt(vx ** 2 + vy ** 2)
    vx -= (Bm / m) * v * vx * dt
    vy -= g * dt + (Bm / m) * v * vy * dt
    return vx, vy


# Plot analytical solution
for theta in range(30, 60, 5):
    theta_rad = math.radians(theta)
    x_max = (v0 ** 2 / g) * math.sin(2 * theta_rad)
    x = [i for i in range(int(x_max) + 1)]
    y = [x_val * math.tan(theta_rad) - (g * x_val ** 2) / (2 * v0 ** 2 * math.cos(theta_rad) ** 2) for x_val in x]
    plt.plot(x, y, label=str(theta) + '° (analytical)')

# Plot Euler method
for theta in range(30, 60, 5):
    theta_rad = math.radians(theta)
    vx, vy = v0 * math.cos(theta_rad), v0 * math.sin(theta_rad)
    x, y = [0], [0]
    dt = 0.1
    while y[-1] >= 0:
        vx, vy = euler_step(vx, vy, dt)
        x.append(x[-1] + vx * dt)
        y.append(y[-1] + vy * dt)
    plt.plot(x, y, '--', label=str(theta) + '° (Euler)')


#only max range:
# theta_rad = math.radians(45)
# x_max = (v0 ** 2 / g) * math.sin(2 * theta_rad)
# x = [i for i in range(int(x_max) + 1)]
# y = [x_val * math.tan(theta_rad) - (g * x_val ** 2) / (2 * v0 ** 2 * math.cos(theta_rad) ** 2) for x_val in x]
# plt.plot(x, y, label=str(45) + '° (analytical)')


# theta_rad1 = math.radians(41)
# vx, vy = v0 * math.cos(theta_rad), v0 * math.sin(theta_rad1)
# x, y = [0], [0]
# dt = 0.1
# while y[-1] >= 0:
#     vx, vy = euler_step(vx, vy, dt)
#     x.append(x[-1] + vx * dt)
#     y.append(y[-1] + vy * dt)
# plt.plot(x, y, '--', label=str(41) + '° (Drag Euler)')

plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.title('Projectile Trajectories')
plt.legend()
plt.savefig('lab_7_3(b).png')
plt.grid()
plt.show()