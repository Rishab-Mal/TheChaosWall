import os
import torch
import torch.optim as optim
from pathlib import Path

from Hamiltonian.model import HNN
from Hamiltonian.data import build_dataloader
import Hamiltonian.config as config


def train_hnn():
    device = torch.device(config.DEVICE)

    # 1. Initialise HNN
    hnn = HNN(
        input_size=config.INPUT_DIM,
        hidden_size=config.HIDDEN_DIM,
    ).to(device)
    optimizer = optim.Adam(hnn.parameters(), lr=config.LR)
    mse_loss = torch.nn.MSELoss()

    # 3. Build DataLoaders
    parquet_path = Path(__file__).resolve().parents[1] / config.PARQUET_PATH
    train_loader, val_loader = build_dataloader(
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

            current_state = seq_in[:, -1, :]  # [B, 4] – last timestep

            pred_derivs = hnn.time_derivatives(current_state)  # [B, 4]
            loss = mse_loss(pred_derivs, target_derivs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{config.EPOCHS} | Train Loss: {avg_loss:.6f}")

        if (epoch + 1) % 50 == 0:
            ckpt = Path(__file__).resolve().parents[1] / f"models/hnn_epoch_{epoch + 1}.pth"
            torch.save(hnn.state_dict(), ckpt)
            print(f"  Saved checkpoint: {ckpt}")

    hnn.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seq_in, target_derivs in val_loader:
            seq_in, target_derivs = seq_in.to(device), target_derivs.to(device)
            current_state = seq_in[:, -1, :]  # [B, 4] – last timestep
            pred_derivs = hnn.time_derivatives(current_state)
            val_loss += mse_loss(pred_derivs, target_derivs).item()
    print(f"Final Val Loss: {val_loss / len(val_loader):.6f}")

    final_path = Path(__file__).resolve().parents[1] / config.HNN_PATH
    torch.save(hnn.state_dict(), final_path)
    print(f"Saved final model: {final_path}")
    return hnn
