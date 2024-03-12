import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the ODE system


def odesys(t, XY):
    x, y = XY
    dxdt = -6 * x / ((1 + x**2)**2) + 2 * y
    dydt = -2 * (x + y) / ((1 + x**2)**2)
    return [dxdt, dydt]

# Define the Lyapunov function


def lyapunov(x, y):
    return x**2 / (1 + x**2) + y**2


# Define time span for simulation
t_span = (0, 10)

# Define initial conditions
initial_conditions = [2, 4]

# Solve the ODE
sol = solve_ivp(odesys, t_span, initial_conditions,
                t_eval=np.linspace(t_span[0], t_span[1], 1000))

# Extract solution
x_solution = sol.y[0]
y_solution = sol.y[1]

# Calculate Lyapunov function along the trajectory
V_solution = lyapunov(x_solution, y_solution)

# Plot trajectory in phase space
plt.figure(figsize=(10, 6))
plt.plot(x_solution, y_solution, label='Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Phase Space Trajectory')
plt.grid(True)
plt.legend()
plt.show()

# Plot Lyapunov function along the trajectory
plt.figure(figsize=(10, 6))
plt.plot(sol.t, V_solution, label='Lyapunov Function')
plt.xlabel('Time')
plt.ylabel('V(x, y)')
plt.title('Lyapunov Function along Trajectory')
plt.grid(True)
plt.legend()
plt.show()

# Contour plot of the Lyapunov function
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = lyapunov(X, Y)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(contour, label='V(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour Plot of Lyapunov Function')
plt.grid(True)
plt.show()
