from pathlib import Path
import torch
from torch import nn
import polars as pl


# Leave blank to set a custom checkpoint path later.
# Example: MODEL_FILEPATH = r"C:\path\to\rnn.pth"
MODEL_FILEPATH = "models/rnn.pth"

# Leave blank to use default parquet in repo root.
# Example: PARQUET_FILEPATH = r"C:\path\to\test_pendulum2.parquet"
PARQUET_FILEPATH = ""


class RNNModel(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 64, output_size: int = 4):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])


def _resolve_checkpoint_path(model_filepath: str = MODEL_FILEPATH) -> Path:
    if model_filepath.strip():
        return Path(model_filepath)
    return Path(__file__).resolve().parents[2] / "models" / "rnn.pth"

def _resolve_parquet_path(parquet_filepath: str = PARQUET_FILEPATH) -> Path:
    if parquet_filepath.strip():
        return Path(parquet_filepath)
    return Path(__file__).resolve().parents[2] / "test_pendulum2.parquet"


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
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()
    return model, target_device


def load_parquet_windows(
    parquet_filepath: str = PARQUET_FILEPATH,
    seq_len: int = 20,
    max_windows: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build sequence/target pairs from parquet organized like test_pendulum2.parquet:
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


if __name__ == "__main__":
    model, device = load_trained_model()
    X, y_true = load_parquet_windows()
    with torch.inference_mode():
        prediction = model(X.to(device)).cpu()
    mse = torch.mean((prediction - y_true) ** 2).item()
    print(f"Loaded model on {device}.")
    print(f"Parquet windows: {tuple(X.shape)} -> predictions: {tuple(prediction.shape)}")
    print(f"MSE on sampled parquet windows: {mse:.6f}")
