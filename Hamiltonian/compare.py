"""
Roll out the trained HNN from the same initial conditions as
comparisons/actual.parquet and save to comparisons/hnn_predicted.parquet.

Run from the project root:
    python Hamiltonian/compare.py
"""

import sys
import torch
import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Hamiltonian.model import HNN        # noqa: E402
import Hamiltonian.config as config     # noqa: E402

CKPT_PATH   = ROOT / "models" / "hnn_epoch_200.pth"
ACTUAL_PATH = ROOT / "comparisons" / "actual.parquet"
OUTPUT_PATH = ROOT / "comparisons" / "hnn_predicted.parquet"


def load_hnn(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    hnn = HNN(input_size=config.INPUT_DIM, hidden_size=config.HIDDEN_DIM)
    hnn.load_state_dict(ckpt["state_dict"])
    hnn.eval()
    state_mean = ckpt["state_mean"]  # [4]
    state_std  = ckpt["state_std"]   # [4]
    return hnn, state_mean, state_std


def _deriv(hnn: HNN, state_mean: torch.Tensor, state_std: torch.Tensor,
           state: torch.Tensor) -> torch.Tensor:
    """Compute HNN time-derivatives for a single state [4]."""
    x = state.unsqueeze(0)                          # [1, 4]
    x_norm = (x - state_mean) / state_std
    d_norm = hnn.time_derivatives(x_norm)           # [1, 4]  (uses autograd internally)
    return (d_norm * state_std).squeeze(0).detach() # [4]


def rollout_rk4(hnn: HNN, state_mean: torch.Tensor, state_std: torch.Tensor,
                initial_state: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Integrate HNN forward with RK4. Returns trajectory [N, 4]."""
    state = torch.tensor(initial_state, dtype=torch.float32)
    trajectory = [initial_state.copy()]

    for i in range(len(times) - 1):
        dt = float(times[i + 1] - times[i])

        k1 = _deriv(hnn, state_mean, state_std, state)
        k2 = _deriv(hnn, state_mean, state_std, state + 0.5 * dt * k1)
        k3 = _deriv(hnn, state_mean, state_std, state + 0.5 * dt * k2)
        k4 = _deriv(hnn, state_mean, state_std, state + dt * k3)

        state = (state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).detach()
        trajectory.append(state.numpy())

    return np.array(trajectory)


if __name__ == "__main__":
    if not ACTUAL_PATH.exists():
        print(f"No actual.parquet found at {ACTUAL_PATH}")
        print("Run:  python project/models/RNNModel.py")
        sys.exit(1)

    if not CKPT_PATH.exists():
        print(f"HNN checkpoint not found: {CKPT_PATH}")
        print("Run:  python Hamiltonian/main.py")
        sys.exit(1)

    hnn, state_mean, state_std = load_hnn(CKPT_PATH)
    print(f"Loaded HNN from {CKPT_PATH.name}")

    df      = pl.read_parquet(str(ACTUAL_PATH)).sort("t")
    feature_cols = ["theta1", "theta2", "theta1_dot", "theta2_dot"]
    actual  = df.select(feature_cols).to_numpy().astype("float32")
    times   = df["t"].to_numpy()

    s0 = actual[0]
    print(f"Initial state: θ₁={s0[0]:.3f}  θ₂={s0[1]:.3f}  "
          f"ω₁={s0[2]:.3f}  ω₂={s0[3]:.3f}")
    print(f"Integrating {len(times)} steps with RK4...")

    traj = rollout_rk4(hnn, state_mean, state_std, s0, times)

    out_df = pl.DataFrame({
        "sim_id":     [0] * len(times),
        "t":          times.tolist(),
        "theta1":     traj[:, 0].tolist(),
        "theta2":     traj[:, 1].tolist(),
        "theta1_dot": traj[:, 2].tolist(),
        "theta2_dot": traj[:, 3].tolist(),
    })
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(str(OUTPUT_PATH))
    print(f"Saved → {OUTPUT_PATH}")
    print("\nNow run:  python project/training/visualize.py")
