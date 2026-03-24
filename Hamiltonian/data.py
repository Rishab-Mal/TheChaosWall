import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def build_dataloader(
    parquet_path: str,
    seq_len: int = 20,
    batch_size: int = 64,
    max_windows: int = 50_000,
    shuffle: bool = True,
) -> tuple:
    """
    Load pendulum parquet data and build (seq_in, target_derivs) pairs.
    seq_in:        [B, seq_len, 4]  – input window of states
    target_derivs: [B, 4]           – time derivatives at the step after the window
                                      estimated via central differences

    Returns (DataLoader, state_mean, state_std) where state_mean/std are [4]
    tensors computed from all training states — used to normalise HNN inputs.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    feature_cols = ["theta1", "theta2", "theta1_dot", "theta2_dot"]

    if HAS_POLARS:
        df = pl.read_parquet(str(path), columns=["sim_id", "t", *feature_cols])
        df = df.sort(["sim_id", "t"])
        groups = list(df.group_by("sim_id", maintain_order=True))
        def get_arrays(group):
            sim_df = group[1]
            arr = sim_df.select(feature_cols).to_numpy().astype("float32")
            times = sim_df["t"].to_numpy().astype("float64")
            return arr, times
    elif HAS_PANDAS:
        df = pd.read_parquet(str(path), columns=["sim_id", "t", *feature_cols])
        df = df.sort_values(["sim_id", "t"])
        groups = list(df.groupby("sim_id"))
        def get_arrays(group):
            sim_df = group[1]
            arr = sim_df[feature_cols].to_numpy().astype("float32")
            times = sim_df["t"].to_numpy().astype("float64")
            return arr, times
    else:
        raise ImportError("Install polars or pandas: pip install polars")

    sequences, targets = [], []

    for group in groups:
        arr, times = get_arrays(group)
        if arr.shape[0] <= seq_len + 1:
            continue

        # Central-difference time derivatives for each timestep
        derivs = np.gradient(arr, times, axis=0).astype("float32")

        for i in range(arr.shape[0] - seq_len - 1):
            sequences.append(arr[i : i + seq_len])
            targets.append(derivs[i + seq_len])
            if len(sequences) >= max_windows:
                break
        if len(sequences) >= max_windows:
            break

    if not sequences:
        raise ValueError("No windows generated. Check parquet data and seq_len.")

    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    y = torch.tensor(np.array(targets), dtype=torch.float32)

    # Normalisation stats
    X_flat = X.reshape(-1, X.shape[-1])          # [N*seq_len, 4]
    state_mean = X_flat.mean(dim=0)              # [4]
    state_std  = X_flat.std(dim=0).clamp(min=1e-6)

    deriv_std = y.std(dim=0).clamp(min=1e-6)     # [4] — per-component deriv scale

    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle), state_mean, state_std, deriv_std


