from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
import polars as pl


# Leave blank to set a custom checkpoint path later.
# Example: MODEL_FILEPATH = r"C:\path\to\rnn.pth"
MODEL_FILEPATH = "models/rnn.pth"

# Leave blank to use default parquet in repo root.
# Example: PARQUET_FILEPATH = r"C:\path\to\test_pendulum2.parquet"
PARQUET_FILEPATH = "Synthetic.parquet"


class RNNModel(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 64, output_size: int = 4):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])


_PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _resolve_checkpoint_path(model_filepath: str = MODEL_FILEPATH) -> Path:
    if not model_filepath.strip():
        return _PROJECT_ROOT / "models" / "rnn.pth"
    p = Path(model_filepath)
    return p if p.is_absolute() else _PROJECT_ROOT / p

def _resolve_parquet_path(parquet_filepath: str = PARQUET_FILEPATH) -> Path:
    if not parquet_filepath.strip():
        return _PROJECT_ROOT / "test_pendulum2.parquet"
    p = Path(parquet_filepath)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def load_trained_model(
    model_filepath: str = MODEL_FILEPATH,
    device: str | None = None,
    input_size: int = 4,
    hidden_size: int = 64,
    output_size: int = 4,
) -> tuple[RNNModel, str]:
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _resolve_checkpoint_path(model_filepath)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = RNNModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    state_dict = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint is incompatible with the current RNNModel (LSTM).\n"
            f"The saved checkpoint at '{checkpoint_path}' was likely trained with a vanilla RNN.\n"
            f"Retrain the RNN using: python project/training/train_rnn.py\n"
            f"Original error: {e}"
        ) from e
    model.to(target_device)
    model.eval()
    return model, target_device


def load_parquet_windows(
    parquet_filepath: str = PARQUET_FILEPATH,
    seq_len: int = 20,
    max_windows: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build sequence/target pairs from parquet:
    columns: sim_id, t, theta1, theta2, theta1_dot, theta2_dot
    """
    parquet_path = _resolve_parquet_path(parquet_filepath)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    feature_cols = ["theta1", "theta2", "theta1_dot", "theta2_dot"]
    required_cols = ["sim_id", "t", *feature_cols]
    df = pl.read_parquet(str(parquet_path), columns=required_cols)
    df = df.sort(["sim_id", "t"])

    sequences = []
    targets = []

    for sim_df in df.group_by("sim_id", maintain_order=True):
        arr = sim_df[1].select(feature_cols).to_numpy().astype("float32")
        if arr.shape[0] <= seq_len:
            continue
        for i in range(arr.shape[0] - seq_len):
            sequences.append(arr[i : i + seq_len])
            targets.append(arr[i + seq_len])
            if len(sequences) >= max_windows:
                break
        if len(sequences) >= max_windows:
            break

    if not sequences:
        raise ValueError("No windows were generated from parquet. Check seq_len and data size.")

    X = torch.tensor(sequences, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    return X, y


# ── Physics simulator (RK4) ────────────────────────────────────────────────────

def _double_pendulum_derivs(
    state: np.ndarray, g: float = 9.81, l1: float = 1.0, l2: float = 1.0,
    m1: float = 1.0, m2: float = 1.0,
) -> np.ndarray:
    """Returns [dtheta1/dt, dtheta2/dt, domega1/dt, domega2/dt]."""
    theta1, theta2, omega1, omega2 = state
    delta = theta2 - theta1
    denom = 2 * m1 + m2 - m2 * np.cos(2 * delta)

    domega1 = (
        -g * (2 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2 * theta2)
        - 2 * np.sin(delta) * m2 * (omega2**2 * l2 + omega1**2 * l1 * np.cos(delta))
    ) / (l1 * denom)

    domega2 = (
        2 * np.sin(delta) * (
            omega1**2 * l1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + omega2**2 * l2 * m2 * np.cos(delta)
        )
    ) / (l2 * denom)

    return np.array([omega1, omega2, domega1, domega2])


def simulate_and_save_parquet(
    theta1: float,
    theta2: float,
    omega1: float,
    omega2: float,
    t_end: float = 10.0,
    dt: float = 0.05,
    output_path: str = "custom_pendulum.parquet",
    sim_id: int = 0,
) -> str:
    """
    Simulate a double pendulum via RK4 and save the trajectory as a parquet file.
    Returns the output path.
    """
    state = np.array([theta1, theta2, omega1, omega2], dtype=np.float64)
    times = np.arange(0.0, t_end + dt, dt)
    trajectory = [state.copy()]

    for _ in range(len(times) - 1):
        k1 = _double_pendulum_derivs(state)
        k2 = _double_pendulum_derivs(state + 0.5 * dt * k1)
        k3 = _double_pendulum_derivs(state + 0.5 * dt * k2)
        k4 = _double_pendulum_derivs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory.append(state.copy())

    traj = np.array(trajectory)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        "sim_id":     [sim_id] * len(times),
        "t":          times.tolist(),
        "theta1":     traj[:, 0].tolist(),
        "theta2":     traj[:, 1].tolist(),
        "theta1_dot": traj[:, 2].tolist(),
        "theta2_dot": traj[:, 3].tolist(),
    })
    df.write_parquet(output_path)
    print(f"Saved {len(times)} timesteps (RK4) → {output_path}")
    return output_path


# ── pendulum.py-based generator (symplectic implicit midpoint) ─────────────────

def generate_from_pendulum_py(
    theta1: float,
    theta2: float,
    T: float = 10.0,
    dt: float = 0.05,
    output_path: str = "comparisons/actual.parquet",
    sim_id: int = 0,
) -> str:
    """
    Generate a trajectory using src/pendulum.py (symplectic implicit midpoint integrator).
    Always starts from rest (theta1_dot = theta2_dot = 0).
    Returns the output path.
    """
    src_path = str(Path(__file__).resolve().parents[2] / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from pendulum import DoublePendulum  # noqa: PLC0415

    sim = DoublePendulum(theta1=theta1, theta2=theta2, T=T, dt=dt)
    df_pd = sim.generateTimeData()

    df = pl.from_pandas(df_pd[["t", "theta1", "theta2", "theta1_dot", "theta2_dot"]])
    df = df.with_columns(pl.lit(sim_id).alias("sim_id"))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"Saved {len(df)} timesteps (pendulum.py symplectic) → {output_path}")
    return output_path


# ── Autoregressive rollout ─────────────────────────────────────────────────────

def autoregressive_rollout(
    actual_parquet: str,
    model: RNNModel,
    device: str,
    output_path: str,
    seq_len: int = 20,
) -> str:
    """
    Seed the RNN with the first seq_len frames from actual_parquet, then predict
    the remaining steps autoregressively. Saves predictions to output_path.
    Returns the output path.
    """
    feature_cols = ["theta1", "theta2", "theta1_dot", "theta2_dot"]
    df = pl.read_parquet(actual_parquet).sort("t")
    actual_arr = df.select(feature_cols).to_numpy().astype("float32")
    time_arr   = df["t"].to_numpy()

    if len(actual_arr) <= seq_len:
        raise ValueError(
            f"Trajectory has only {len(actual_arr)} steps — need > {seq_len}. "
            "Increase T or reduce dt."
        )

    window = actual_arr[:seq_len].copy()
    n_steps = len(actual_arr) - seq_len
    predictions = []

    model.eval()
    with torch.inference_mode():
        for _ in range(n_steps):
            x   = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
            out = model(x).cpu().numpy()[0]

            # Unwrap predicted angles so they stay continuous with the previous step.
            # Without this, a jump from 6.27 → 0.01 looks like a 6.26-rad error
            # but is physically ~0 (2π wrap-around).
            prev = window[-1, :2]
            for j in range(2):
                delta = out[j] - prev[j]
                out[j] = prev[j] + (delta + np.pi) % (2 * np.pi) - np.pi

            predictions.append(out)
            window = np.vstack([window[1:], out])

    predictions = np.array(predictions)  # (n_steps, 4)
    pred_times  = time_arr[seq_len:]

    pred_df = pl.DataFrame({
        "sim_id":     [0] * n_steps,
        "t":          pred_times.tolist(),
        "theta1":     predictions[:, 0].tolist(),
        "theta2":     predictions[:, 1].tolist(),
        "theta1_dot": predictions[:, 2].tolist(),
        "theta2_dot": predictions[:, 3].tolist(),
    })
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pred_df.write_parquet(output_path)
    print(f"Saved {n_steps} RNN predictions → {output_path}")
    return output_path


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def _angular_diff(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Shortest signed angular difference in [-π, π]."""
    return torch.atan2(torch.sin(pred - true), torch.cos(pred - true))


def _regression_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, circular: bool = False) -> dict:
    err = _angular_diff(y_pred, y_true) if circular else (y_pred - y_true)
    mse = torch.mean(err ** 2).item()
    rmse = mse ** 0.5
    mae = torch.mean(torch.abs(err)).item()
    ss_res = torch.sum(err ** 2)
    y_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    r2 = (1.0 - (ss_res / (ss_tot + 1e-12))).item()
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def _print_evaluation(y_pred: torch.Tensor, y_true: torch.Tensor, X: torch.Tensor) -> None:
    feature_names = ["theta1", "theta2", "theta1_dot", "theta2_dot"]

    print("\nEvaluation Summary")
    print(f"  Num windows:     {X.shape[0]}")
    print(f"  Sequence shape:  {tuple(X.shape)}")
    print(f"  Target shape:    {tuple(y_true.shape)}")

    overall = _regression_metrics(y_pred, y_true)
    print("\nOverall Metrics")
    print(
        f"  MSE: {overall['mse']:.6f} | RMSE: {overall['rmse']:.6f} | "
        f"MAE: {overall['mae']:.6f} | R2: {overall['r2']:.6f}"
    )

    print("\nPer-Feature Metrics")
    for i, name in enumerate(feature_names):
        is_angle = i < 2  # theta1, theta2 are circular
        m = _regression_metrics(y_pred[:, i:i+1], y_true[:, i:i+1], circular=is_angle)
        tag = " (circular)" if is_angle else ""
        print(
            f"  {name:12s}  MSE: {m['mse']:.6f} | RMSE: {m['rmse']:.6f} | "
            f"MAE: {m['mae']:.6f} | R2: {m['r2']:.6f}{tag}"
        )

    baseline_pred = X[:, -1, :]
    baseline = _regression_metrics(baseline_pred, y_true)
    print("\nBaseline (predict next = last input timestep)")
    print(
        f"  MSE: {baseline['mse']:.6f} | RMSE: {baseline['rmse']:.6f} | "
        f"MAE: {baseline['mae']:.6f} | R2: {baseline['r2']:.6f}"
    )

    print("\nSample Predictions (first 5)")
    max_rows = min(5, y_true.shape[0])
    for i in range(max_rows):
        print(f"  idx {i:3d} | true={[f'{v:.4f}' for v in y_true[i].tolist()]} "
              f"| pred={[f'{v:.4f}' for v in y_pred[i].tolist()]}")


# ── Entry point ────────────────────────────────────────────────────────────────

def _prompt_float(prompt: str, default: float) -> float:
    raw = input(f"  {prompt} [{default}]: ").strip()
    return float(raw) if raw else default


if __name__ == "__main__":
    print("=== Double Pendulum RNN ===")
    print("  (Simulation starts from rest: θ̇₁ = θ̇₂ = 0)")
    print("\nEnter initial conditions:")
    theta1 = _prompt_float("theta1 (rad)", 1.0)
    theta2 = _prompt_float("theta2 (rad)", 0.5)

    print("\nSimulation parameters:")
    T  = _prompt_float("Duration (s) ", 10.0)
    dt = _prompt_float("Timestep (s) ", 0.05)

    root            = Path(__file__).resolve().parents[2]
    comparisons_dir = root / "comparisons"
    actual_path     = str(comparisons_dir / "actual.parquet")
    pred_path       = str(comparisons_dir / "rnn_predicted.parquet")

    # 1. Generate actual trajectory using pendulum.py (symplectic integrator)
    print()
    generate_from_pendulum_py(theta1, theta2, T, dt, actual_path)

    # 2. Load trained RNN
    model, device = load_trained_model()
    print(f"RNN loaded on: {device}")

    # 3. Autoregressive rollout → comparisons/rnn_predicted.parquet
    print("\nRunning autoregressive rollout...")
    autoregressive_rollout(actual_path, model, device, pred_path)

    # 4. Single-step evaluation metrics
    print("\nSingle-step evaluation on generated trajectory:")
    X, y_true = load_parquet_windows(parquet_filepath=actual_path)
    with torch.inference_mode():
        single_preds = model(X.to(device)).cpu()
    _print_evaluation(single_preds, y_true, X)

    # 5. Launch comparison visualization
    viz_path = Path(__file__).resolve().parents[1] / "training" / "visualize.py"
    print(f"\nLaunching visualization: {viz_path}")
    import subprocess
    subprocess.run([sys.executable, str(viz_path)], check=True)
