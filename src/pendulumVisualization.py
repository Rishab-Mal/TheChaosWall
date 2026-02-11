import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. SETUP PARAMETERS ---
m1, m2 = 1.0, 1.0
l1, l2 = 1.0, 1.0
g = 9.81
dt = 0.01  # Time step

# Initial state: [theta1, theta2, theta1_dot, theta2_dot]
state = np.array([np.pi/2, np.pi/2, 0.0, 0.0])

def get_derivs(state):
    """Calculates the angular accelerations (the math from your code)."""
    t1, t2, w1, w2 = state
    cos_delta = np.cos(t1 - t2)
    sin_delta = np.sin(t1 - t2)
    
    den1 = (m1 + m2) * l1 - m2 * l1 * cos_delta**2
    t1_ddot = (m2 * l1 * w1**2 * sin_delta * cos_delta
               + m2 * g * np.sin(t2) * cos_delta
               + m2 * l2 * w2**2 * sin_delta
               - (m1 + m2) * g * np.sin(t1)) / den1

    den2 = (l2 / l1) * den1
    t2_ddot = (-m2 * l2 * w2**2 * sin_delta * cos_delta
               + (m1 + m2) * (g * np.sin(t1) * cos_delta
               - l1 * w1**2 * sin_delta
               - g * np.sin(t2))) / den2
    
    return np.array([w1, w2, t1_ddot, t2_ddot])

# --- 2. THE ANIMATION ENGINE ---
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2, color='#1f77b4', markersize=8)
trace, = ax.plot([], [], '-', lw=1, color='gray', alpha=0.5)
history_x, history_y = [], []

def update(frame):
    global state
    # Update state using the derivatives
    state += get_derivs(state) * dt
    
    # Convert angles to (x, y) coordinates
    x1 = l1 * np.sin(state[0])
    y1 = -l1 * np.cos(state[0])
    x2 = x1 + l2 * np.sin(state[1])
    y2 = y1 - l2 * np.cos(state[1])
    
    # Update the pendulum rods
    line.set_data([0, x1, x2], [0, y1, y2])
    
    # Update the "ghost" trail
    history_x.append(x2)
    history_y.append(y2)
    trace.set_data(history_x[-50:], history_y[-50:]) # Keep last 50 points
    
    return line, trace

ani = FuncAnimation(fig, update, frames=200, interval=20, blit=True)
plt.show()