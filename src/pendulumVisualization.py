# //src/pendulumVisualization.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from pendulum import DoublePendulum


# -----------------------------------------
# 1. Simulation Parameters
# -----------------------------------------

l1 = 1.0
l2 = 1.0
dt = 0.01
T = 5.0   # longer for better visual effect

# Initial angles (radians)
theta1_init = np.pi / 2
theta2_init = np.pi / 2 + 0.01  # small offset to show chaos

pendulum = DoublePendulum(
    theta1=theta1_init,
    theta2=theta2_init,
    l1=l1,
    l2=l2,
    dt=dt,
    T=T
)

df = pendulum.generateTimeData()

theta1_vals = df["theta1"].values
theta2_vals = df["theta2"].values

# -----------------------------------------
# 2. Setup Figure
# -----------------------------------------

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect("equal")
ax.grid(True)
ax.set_title("Double Pendulum (RK45 Integration)")

line, = ax.plot([], [], "o-", lw=2)
trace, = ax.plot([], [], "-", lw=1, alpha=0.5)

history_x, history_y = [], []


# -----------------------------------------
# 3. Animation Update
# -----------------------------------------

def update(frame):

    theta1 = theta1_vals[frame]
    theta2 = theta2_vals[frame]

    # Convert to Cartesian
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    # Update rods
    line.set_data([0, x1, x2], [0, y1, y2])

    # Update trail
    history_x.append(x2)
    history_y.append(y2)
    trace.set_data(history_x[-150:], history_y[-150:])

    return line, trace


# -----------------------------------------
# 4. Run Animation
# -----------------------------------------

ani = FuncAnimation(
    fig,
    update,
    frames=len(theta1_vals),
    interval=dt * 1000,
    blit=True
)

plt.show()
