import os
import sys
import torch
import torch.optim as optim
from pathlib import Path

# Allow importing RNNModel from project/models/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Hamiltonian.model import HNN           # noqa: E402
from Hamiltonian.data import build_dataloader  # noqa: E402
import Hamiltonian.config as config         # noqa: E402
from project.models.RNNModel import RNNModel  # noqa: E402


def train_hnn():
    device = torch.device(config.DEVICE)

    # 1. Load pre-trained RNN (frozen – used only to map sequences → current state)
    rnn_path = Path(__file__).resolve().parents[1] / config.RNN_PATH
    if not rnn_path.exists():
        raise FileNotFoundError(
            f"RNN checkpoint not found: {rnn_path}\n"
            "Train the RNN first: python project/training/train_rnn.py"
        )
    # RNN stays on CPU — LSTM is not supported on DirectML
    rnn = RNNModel(
        input_size=config.INPUT_DIM,
        hidden_size=config.RNN_HIDDEN,
        output_size=config.INPUT_DIM,
    )
    rnn.load_state_dict(torch.load(rnn_path, map_location="cpu", weights_only=True))
    rnn.eval()

    # 2. Initialise HNN
    hnn = HNN(
        input_size=config.INPUT_DIM,
        hidden_size=config.HIDDEN_DIM,
    ).to(device)
    optimizer = optim.Adam(hnn.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-6)
    mse_loss = torch.nn.MSELoss()

    # 3. Build DataLoader + normalisation stats
    parquet_path = Path(__file__).resolve().parents[1] / config.PARQUET_PATH
    train_loader, state_mean, state_std, deriv_std = build_dataloader(
        parquet_path=str(parquet_path),
        seq_len=config.SEQ_LEN,
        batch_size=config.BATCH_SIZE,
    )
    state_mean = state_mean.to(device)
    state_std  = state_std.to(device)
    deriv_std  = deriv_std.to(device)
    print(f"Deriv std (per component): {deriv_std.tolist()}")

    os.makedirs(Path(__file__).resolve().parents[1] / "models", exist_ok=True)

    for epoch in range(config.EPOCHS):
        hnn.train()
        total_loss = 0.0

        for seq_in, target_derivs in train_loader:
            target_derivs = target_derivs.to(device)

            with torch.no_grad():
                current_state = rnn(seq_in)  # [B, 4] — runs on CPU
            current_state = current_state.to(device)

            # Normalise input so HNN sees O(1) values (better for tanh)
            state_norm = (current_state - state_mean) / state_std
            pred_derivs_norm = hnn.time_derivatives(state_norm)  # [B, 4] in normalised space

            # Un-normalise: d(x)/dt = d(x̃)/dt * σ
            pred_derivs = pred_derivs_norm * state_std

            # Normalise each component by its own std so all 4 contribute equally
            loss = mse_loss(pred_derivs / deriv_std, target_derivs / deriv_std)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{config.EPOCHS} | HNN Physics Loss: {avg_loss:.6f}  lr={lr:.2e}")

        if (epoch + 1) % 50 == 0:
            ckpt = Path(__file__).resolve().parents[1] / f"models/hnn_epoch_{epoch + 1}.pth"
            torch.save({
                "state_dict": hnn.state_dict(),
                "state_mean": state_mean.cpu(),
                "state_std":  state_std.cpu(),
                "deriv_std":  deriv_std.cpu(),
            }, ckpt)
            print(f"  Saved checkpoint: {ckpt}")

    return hnn
