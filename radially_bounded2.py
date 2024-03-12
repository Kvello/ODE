import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the ODEs


def odes(t, z):
    x, y = z
    dxdt = -6*x/((1+x**2)**2) + 2*y
    dydt = -2*(x+y)/((1+x**2)**2)
    return [dxdt, dydt]

# Define the Lyapunov function


def lyapunov(x, y):
    return x**2 / (1 + x**2) + y**2


# Define initial conditions
num_ICs = 10
x_initial = np.linspace(-4, 4, num=num_ICs)
y_initial = np.linspace(-4, 4, num=num_ICs)
X_initial, Y_initial = np.meshgrid(x_initial, y_initial)
initial_conditions_grid = np.vstack(
    (X_initial.flatten(), Y_initial.flatten())).T
T = 100  # Simulation time
dt = 1e-2  # Time step
special_initial_conditions = [(1, 1), (2, 4)]  # Special initial conditions

# Create figure with subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Phase plane plot with trajectories
axs[0].set_title('Phase Plane Plot')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_xlim(-5, 5)
axs[0].set_ylim(-5, 5)
axs[0].grid(True)  # Add grid

# Plot Lyapunov function contour
x_range = np.linspace(-4, 4, 1000)
y_range = np.linspace(-4, 4, 1000)
X, Y = np.meshgrid(x_range, y_range)
Z = lyapunov(X, Y)
axs[1].contour(X, Y, Z, levels=40, cmap='viridis')
axs[1].set_title('Lyapunov Function Contour')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].grid(True)  # Add grid

# Subplot 2: Lyapunov function evolution for special initial conditions
axs[2].set_title('Lyapunov Function Evolution')

# Iterate over initial conditions and plot trajectories
for ic in initial_conditions_grid:
    sol = solve_ivp(odes, [0, T], ic, t_eval=np.arange(0, T, dt))
    axs[0].plot(sol.y[0], sol.y[1], color='blue', alpha=0.5)
    axs[0].plot(ic[0], ic[1], 'o', color='blue', alpha=0.5)

# Plot trajectories for special initial conditions
special_colors = ['red', 'green']
for i, ic in enumerate(special_initial_conditions):
    sol_special = solve_ivp(odes, [0, T], ic, t_eval=np.arange(0, T, dt))
    axs[0].plot(sol_special.y[0], sol_special.y[1],
                color=special_colors[i], label=f"Special IC {i+1}", linestyle='--')
    axs[0].plot(ic[0], ic[1], 'o', color=special_colors[i])

    # Plot Lyapunov function evolution
    lyapunov_values = lyapunov(sol_special.y[0], sol_special.y[1])
    axs[2].plot(sol_special.t, lyapunov_values,
                label=f"Special IC {i+1}", color=special_colors[i])

# Add legend to subplot 0
axs[0].legend()

# Add legend to subplot 2
axs[2].legend()
axs[2].set_xlabel('Time')
axs[2].set_ylabel('V(x, y)')
axs[2].grid(True)  # Add grid

plt.tight_layout()
plt.show()
