import os
import torch
import torch.optim as optim
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Hamiltonian.model import HNN
from Hamiltonian.data import build_direct_dataloader
import Hamiltonian.config as config


def train_hnn():
    device = torch.device(config.DEVICE)

    hnn = HNN(
        input_size=config.INPUT_DIM,
        hidden_size=config.HIDDEN_DIM,
    ).to(device)

    optimizer = optim.Adam(hnn.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=15, min_lr=1e-6
    )
    mse_loss = torch.nn.MSELoss()

    parquet_path = Path(__file__).resolve().parents[1] / config.PARQUET_PATH
    train_loader, state_mean, state_std, deriv_std = build_direct_dataloader(
        parquet_path=str(parquet_path),
        batch_size=config.BATCH_SIZE,
    )
    state_mean = state_mean.to(device)
    state_std  = state_std.to(device)
    deriv_std  = deriv_std.to(device)

    print(f"Training on {len(train_loader.dataset):,} state-derivative pairs")
    print(f"Device: {device}  |  hidden_size: {config.HIDDEN_DIM}  |  epochs: {config.EPOCHS}")
    print(f"Deriv std: {[f'{v:.4f}' for v in deriv_std.tolist()]}")

    os.makedirs(Path(__file__).resolve().parents[1] / "models", exist_ok=True)

    for epoch in range(config.EPOCHS):
        hnn.train()
        total_loss = 0.0

        for state, target_derivs in train_loader:
            state         = state.to(device)
            target_derivs = target_derivs.to(device)

            state_norm       = (state - state_mean) / state_std
            pred_derivs_norm = hnn.time_derivatives(state_norm)

            # Un-normalise back to physical units then scale by deriv_std
            # so all 4 components contribute equally to the loss
            pred_derivs = pred_derivs_norm * state_std
            loss = mse_loss(pred_derivs / deriv_std, target_derivs / deriv_std)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1:>4}/{config.EPOCHS} | Loss: {avg_loss:.6f}  lr={lr:.2e}")

        if (epoch + 1) % 50 == 0:
            ckpt = Path(__file__).resolve().parents[1] / f"models/hnn_epoch_{epoch + 1}.pth"
            torch.save({
                "state_dict": hnn.state_dict(),
                "state_mean": state_mean.cpu(),
                "state_std":  state_std.cpu(),
                "deriv_std":  deriv_std.cpu(),
            }, ckpt)
            print(f"  Saved → {ckpt}")

    return hnn
