# //src/pendulumVisualization.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# -----------------------------------------
# 1. Load Data
# -----------------------------------------

data_path = "test_pendulum.parquet"
df = pd.read_parquet(data_path)

required_cols = {"theta1", "theta2"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns in {data_path}: {sorted(missing_cols)}")

theta1_vals = df["theta1"].to_numpy()
theta2_vals = df["theta2"].to_numpy()
num_frames = len(theta1_vals)

# Use values from parquet if present; otherwise fallback defaults.
l1 = float(df["l1"].iloc[0]) if "l1" in df.columns else 1.0
l2 = float(df["l2"].iloc[0]) if "l2" in df.columns else 1.0
dt = float(df["dt"].iloc[0]) if "dt" in df.columns else 0.01
m1 = float(df["m1"].iloc[0]) if "m1" in df.columns else 1.0
m2 = float(df["m2"].iloc[0]) if "m2" in df.columns else 1.0
g = float(df["g"].iloc[0]) if "g" in df.columns else 9.81

if "t" in df.columns:
    t_vals = df["t"].to_numpy()
else:
    t_vals = np.arange(num_frames) * dt

if "theta1_dot" in df.columns and "theta2_dot" in df.columns:
    theta1_dot_vals = df["theta1_dot"].to_numpy()
    theta2_dot_vals = df["theta2_dot"].to_numpy()
else:
    # Unwrap for derivative estimation to avoid 2*pi jumps.
    theta1_unwrapped = np.unwrap(theta1_vals)
    theta2_unwrapped = np.unwrap(theta2_vals)
    theta1_dot_vals = np.gradient(theta1_unwrapped, t_vals)
    theta2_dot_vals = np.gradient(theta2_unwrapped, t_vals)

# Total mechanical energy at each sample.
delta = theta1_vals - theta2_vals
v1_sq = (l1 * theta1_dot_vals) ** 2
v2_sq = (
    (l1 * theta1_dot_vals) ** 2
    + (l2 * theta2_dot_vals) ** 2
    + 2.0 * l1 * l2 * theta1_dot_vals * theta2_dot_vals * np.cos(delta)
)
kinetic = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
potential = -(m1 + m2) * g * l1 * np.cos(theta1_vals) - m2 * g * l2 * np.cos(theta2_vals)
energy_total = kinetic + potential
energy0 = energy_total[0]
energy_rel_drift = (energy_total - energy0) / (abs(energy0) + 1e-12)
max_abs_drift_pct = 100.0 * float(np.max(np.abs(energy_rel_drift)))
rms_drift_pct = 100.0 * float(np.sqrt(np.mean(energy_rel_drift ** 2)))
print(f"Energy check: max |DeltaE/E0| = {max_abs_drift_pct:.6f}%")
print(f"Energy check: RMS DeltaE/E0 = {rms_drift_pct:.6f}%")

# -----------------------------------------
# 2. Setup Figure
# -----------------------------------------

fig, (ax, ax_energy) = plt.subplots(
    nrows=2,
    figsize=(7, 8),
    gridspec_kw={"height_ratios": [4, 1.4]}
)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect("equal")
ax.grid(True)
ax.set_title("Double Pendulum (Parquet Data)")

line, = ax.plot([], [], "o-", lw=2)
trace, = ax.plot([], [], "-", lw=1, alpha=0.5)
energy_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10,
    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"}
)

ax_energy.plot(t_vals, energy_total, lw=1.5, color="tab:blue")
energy_marker, = ax_energy.plot([], [], "o", color="tab:red", ms=5)
ax_energy.set_xlim(t_vals[0], t_vals[-1] if len(t_vals) > 1 else t_vals[0] + dt)
e_min = float(np.min(energy_total))
e_max = float(np.max(energy_total))
if np.isclose(e_min, e_max):
    pad = max(1e-6, 0.05 * max(1.0, abs(e_min)))
    ax_energy.set_ylim(e_min - pad, e_max + pad)
else:
    pad = 0.1 * (e_max - e_min)
    ax_energy.set_ylim(e_min - pad, e_max + pad)
ax_energy.set_title("Total Mechanical Energy")
ax_energy.set_xlabel("Time [s]")
ax_energy.set_ylabel("Energy [J]")
ax_energy.grid(True, alpha=0.3)
ax_energy.axhline(energy0, color="tab:green", ls="--", lw=1, alpha=0.8)
ax_energy.text(
    0.01, 0.98,
    f"max |DeltaE/E0| = {max_abs_drift_pct:.4f}%\nRMS DeltaE/E0 = {rms_drift_pct:.4f}%",
    transform=ax_energy.transAxes,
    va="top",
    fontsize=9,
    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"}
)

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

    E = energy_total[frame]
    drift_pct = 100.0 * (E - energy0) / (abs(energy0) + 1e-12)
    energy_text.set_text(f"E = {E:.6f} J\nDeltaE/E0 = {drift_pct:+.4f}%")
    energy_marker.set_data([t_vals[frame]], [E])

    return line, trace, energy_text, energy_marker


# -----------------------------------------
# 4. Run Animation
# -----------------------------------------

ani = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=dt * 1000,
    blit=True
)

plt.show()
