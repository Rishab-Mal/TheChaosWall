import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DATA_PATH   = "test_pendulum2.parquet"
ANIMATE_SIM = -1   # index into unique_sims list; -1 = last (most chaotic)

# -----------------------------------------
# 1. Load & compute energy for every sim
# -----------------------------------------

df_all = pd.read_parquet(DATA_PATH)
unique_sims = df_all["sim_id"].unique()


def compute_energy(df):
    m1 = float(df["m1"].iloc[0]); m2 = float(df["m2"].iloc[0])
    l1 = float(df["l1"].iloc[0]); l2 = float(df["l2"].iloc[0])
    g  = float(df["g"].iloc[0])
    t1 = df["theta1"].to_numpy();     t2 = df["theta2"].to_numpy()
    o1 = df["theta1_dot"].to_numpy(); o2 = df["theta2_dot"].to_numpy()
    delta = t1 - t2
    KE = (0.5*(m1+m2)*l1**2*o1**2
          + m2*l1*l2*o1*o2*np.cos(delta)
          + 0.5*m2*l2**2*o2**2)
    PE = -(m1+m2)*g*l1*np.cos(t1) - m2*g*l2*np.cos(t2)
    return KE + PE


# -----------------------------------------
# 2. Static figure: energy drift for all sims
# -----------------------------------------

fig_e, ax_e = plt.subplots(figsize=(11, 4))
ax_e.set_title("Energy Conservation — All Starting Conditions  (symplectic implicit midpoint)")
ax_e.set_xlabel("Time [s]")
ax_e.set_ylabel("Relative drift  (E − E₀) / |E₀|")
ax_e.axhline(0, color="k", lw=0.5)

for sid in unique_sims:
    sub = df_all[df_all["sim_id"] == sid].copy()
    label = sub["label"].iloc[0] if "label" in sub.columns else sid
    t1d   = sub["init_theta1_deg"].iloc[0] if "init_theta1_deg" in sub.columns else "?"
    t2d   = sub["init_theta2_deg"].iloc[0] if "init_theta2_deg" in sub.columns else "?"
    t_arr = sub["t"].to_numpy()
    E     = compute_energy(sub)
    E0    = E[0]
    m1_s = float(sub["m1"].iloc[0]); m2_s = float(sub["m2"].iloc[0])
    l1_s = float(sub["l1"].iloc[0]); l2_s = float(sub["l2"].iloc[0])
    g_s  = float(sub["g"].iloc[0])
    E_scale = max(abs(E0), (m1_s + m2_s) * g_s * l1_s + m2_s * g_s * l2_s)
    drift = (E - E0) / E_scale
    ax_e.plot(t_arr, drift, lw=1.2, label=f"{label}  ({t1d}°, {t2d}°)")

ax_e.legend(fontsize=7, ncol=2, loc="upper left")
ax_e.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.tight_layout()
plt.show(block=False)   # keep open while animation loads

# -----------------------------------------
# 3. Pick the sim to animate
# -----------------------------------------

anim_sid = unique_sims[ANIMATE_SIM]
df = df_all[df_all["sim_id"] == anim_sid].copy().reset_index(drop=True)

l1 = float(df["l1"].iloc[0]) if "l1" in df.columns else 1.0
l2 = float(df["l2"].iloc[0]) if "l2" in df.columns else 1.0
m1 = float(df["m1"].iloc[0]) if "m1" in df.columns else 1.0
m2 = float(df["m2"].iloc[0]) if "m2" in df.columns else 1.0
g  = float(df["g"].iloc[0])  if "g"  in df.columns else 9.81
dt = float(df["dt"].iloc[0]) if "dt" in df.columns else 0.01
label_str = df["label"].iloc[0] if "label" in df.columns else anim_sid

theta1_vals     = df["theta1"].to_numpy()
theta2_vals     = df["theta2"].to_numpy()
theta1_dot_vals = df["theta1_dot"].to_numpy()
theta2_dot_vals = df["theta2_dot"].to_numpy()
t_vals          = df["t"].to_numpy()
num_frames      = len(theta1_vals)

energy_total = compute_energy(df)
energy0      = energy_total[0]
E_scale      = max(abs(energy0), (m1 + m2) * g * l1 + m2 * g * l2)
rel_drift    = (energy_total - energy0) / E_scale
max_drift_pct = 100.0 * float(np.max(np.abs(rel_drift)))

# -----------------------------------------
# 4. Animation figure
# -----------------------------------------

fig, (ax_anim, ax_en) = plt.subplots(
    nrows=2, figsize=(7, 8),
    gridspec_kw={"height_ratios": [4, 1.5]}
)

lim = (l1 + l2) * 1.1
ax_anim.set_xlim(-lim, lim)
ax_anim.set_ylim(-lim, lim)
ax_anim.set_aspect("equal")
ax_anim.grid(True, alpha=0.3)
ax_anim.set_title(
    f"Double Pendulum — {label_str}  "
    f"(θ₁={df['init_theta1_deg'].iloc[0]:.0f}°, θ₂={df['init_theta2_deg'].iloc[0]:.0f}°)\n"
    f"max |ΔE/E₀| = {max_drift_pct:.2e}%",
    fontsize=9
)

line,  = ax_anim.plot([], [], "o-", lw=2, color="#2c3e50", mfc="#e74c3c", ms=8)
trace, = ax_anim.plot([], [], "-",  lw=1, alpha=0.4, color="#3498db")
info   = ax_anim.text(
    0.02, 0.95, "", transform=ax_anim.transAxes, family="monospace",
    fontsize=8, va="top",
    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"}
)

# Energy subplot
ax_en.plot(t_vals, rel_drift, lw=1.2, color="#2980b9")
ax_en.axhline(0, color="k", lw=0.5)
e_marker, = ax_en.plot([], [], "o", color="#e74c3c", ms=5)
ax_en.set_xlim(t_vals[0], t_vals[-1])
ax_en.set_ylabel("ΔE / |E₀|")
ax_en.set_xlabel("Time [s]")
ax_en.grid(True, alpha=0.2)
ax_en.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# Auto-scale y so tiny fluctuations are visible
e_std = np.std(rel_drift)
margin = max(e_std * 3, 1e-10)
ax_en.set_ylim(-margin, margin)

history_x, history_y = [], []

def update(frame):
    t1 = theta1_vals[frame]
    t2 = theta2_vals[frame]
    x1, y1 =  l1*np.sin(t1),              -l1*np.cos(t1)
    x2, y2 =  x1 + l2*np.sin(t2),   y1 - l2*np.cos(t2)

    line.set_data([0, x1, x2], [0, y1, y2])
    history_x.append(x2)
    history_y.append(y2)
    trace.set_data(history_x[-150:], history_y[-150:])

    d = 100.0 * rel_drift[frame]
    info.set_text(
        f"t = {t_vals[frame]:.2f} s\n"
        f"E = {energy_total[frame]:.6f} J\n"
        f"ΔE/E₀ = {d:+.3e}%"
    )
    e_marker.set_data([t_vals[frame]], [rel_drift[frame]])
    return line, trace, info, e_marker

ani = FuncAnimation(fig, update, frames=num_frames,
                    interval=dt * 1000, blit=True)
plt.tight_layout()
plt.show()
