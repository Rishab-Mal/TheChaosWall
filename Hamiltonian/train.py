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
            "Train the RNN first (project/training/train.py)."
        )
    rnn = RNNModel(
        input_size=config.INPUT_DIM,
        hidden_size=config.RNN_HIDDEN,
        output_size=config.INPUT_DIM,
    ).to(device)
    rnn.load_state_dict(torch.load(rnn_path, map_location=device, weights_only=True))
    rnn.eval()

    # 2. Initialise HNN
    hnn = HNN(
        input_size=config.INPUT_DIM,
        hidden_size=config.HIDDEN_DIM,
    ).to(device)
    optimizer = optim.Adam(hnn.parameters(), lr=config.LR)
    mse_loss = torch.nn.MSELoss()

    # 3. Build DataLoader
    parquet_path = Path(__file__).resolve().parents[1] / config.PARQUET_PATH
    train_loader = build_dataloader(
        parquet_path=str(parquet_path),
        seq_len=config.SEQ_LEN,
        batch_size=config.BATCH_SIZE,
    )

    os.makedirs(Path(__file__).resolve().parents[1] / "models", exist_ok=True)

    for epoch in range(config.EPOCHS):
        hnn.train()
        total_loss = 0.0

        for seq_in, target_derivs in train_loader:
            seq_in, target_derivs = seq_in.to(device), target_derivs.to(device)

            with torch.no_grad():
                current_state = rnn(seq_in)  # [B, 4]

            pred_derivs = hnn.time_derivatives(current_state)  # [B, 4]
            loss = mse_loss(pred_derivs, target_derivs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{config.EPOCHS} | HNN Physics Loss: {avg_loss:.6f}")

        if (epoch + 1) % 50 == 0:
            ckpt = Path(__file__).resolve().parents[1] / f"models/hnn_epoch_{epoch + 1}.pth"
            torch.save(hnn.state_dict(), ckpt)
            print(f"  Saved checkpoint: {ckpt}")

    return hnn
