"""
Train LSTM on test_pendulum2.parquet, then roll out predictions
autoregressively on a held-out simulation and plot side by side.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.layers import build_rnn

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'test_pendulum2.parquet')
SEQ_LEN      = 20
N_FEATURES   = 4
HIDDEN_SIZE  = 64
BATCH_SIZE   = 64
EPOCHS       = 30
LR           = 1e-3
FEATURE_COLS = ['theta1', 'theta2', 'theta1_dot', 'theta2_dot']
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Model ─────────────────────────────────────────────────────────────────────
class RNNWithHead(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type='LSTM'):
        super().__init__()
        self.rnn = build_rnn(rnn_type, input_size=input_size, hidden_size=hidden_size)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])

# ── Load data ─────────────────────────────────────────────────────────────────
df       = pl.read_parquet(PARQUET_PATH)
sim_ids  = df['sim_id'].unique().to_list()

# Hold out last sim for evaluation
test_sim_id  = sim_ids[-1]
train_sim_ids = sim_ids[:-1]

def build_windows(data_df, sim_id_list, seq_len):
    """Slide a window of seq_len over each simulation to build (X, y) pairs."""
    Xs, ys = [], []
    for sid in sim_id_list:
        vals = (
            data_df.filter(pl.col('sim_id') == sid)
            .sort('t')
            .select(FEATURE_COLS)
            .to_numpy()
            .astype('float32')
        )
        for i in range(len(vals) - seq_len):
            Xs.append(vals[i : i + seq_len])
            ys.append(vals[i + seq_len])
    return np.array(Xs), np.array(ys)

print("Building training windows...")
X_train, y_train = build_windows(df, train_sim_ids, SEQ_LEN)

# ── Normalization (fit on training data only) ─────────────────────────────────
mean = X_train.reshape(-1, N_FEATURES).mean(axis=0)
std  = X_train.reshape(-1, N_FEATURES).std(axis=0) + 1e-8  # avoid div-by-zero

X_train = (X_train - mean) / std
y_train = (y_train - mean) / std

X_t = torch.tensor(X_train)
y_t = torch.tensor(y_train)

dataset     = torch.utils.data.TensorDataset(X_t, y_t)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ── Train ─────────────────────────────────────────────────────────────────────
model     = RNNWithHead(N_FEATURES, HIDDEN_SIZE, N_FEATURES, rnn_type='LSTM').to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print(f"Training on {DEVICE} for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.5f}")

# ── Autoregressive rollout on held-out sim ────────────────────────────────────
test_vals = (
    df.filter(pl.col('sim_id') == test_sim_id)
    .sort('t')
    .select(FEATURE_COLS)
    .to_numpy()
    .astype('float32')
)
test_t = (
    df.filter(pl.col('sim_id') == test_sim_id)
    .sort('t')
    ['t'].to_numpy()
)

# Seed with first SEQ_LEN real frames, then predict autoregressively
seed    = test_vals[:SEQ_LEN].copy()   # (SEQ_LEN, 4)
n_steps = len(test_vals) - SEQ_LEN

# Normalize seed using training stats
seed_norm = (seed - mean) / std

model.eval()
window      = seed_norm.copy()         # sliding window (normalized)
predictions = []

with torch.inference_mode():
    for _ in range(n_steps):
        x   = torch.tensor(window[np.newaxis]).to(DEVICE)  # (1, SEQ_LEN, 4)
        out = model(x).cpu().numpy()[0]                     # (4,) normalized
        predictions.append(out)
        window = np.vstack([window[1:], out])               # slide forward

# Denormalize predictions back to original scale
predictions = np.array(predictions) * std + mean  # (n_steps, 4)
actual      = test_vals[SEQ_LEN:]                 # (n_steps, 4) already raw
time_axis   = test_t[SEQ_LEN:]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(f'LSTM vs Actual — sim: {test_sim_id}', fontsize=13)

for i, (ax, name) in enumerate(zip(axes.flat, FEATURE_COLS)):
    ax.plot(time_axis, actual[:, i],      'g',  lw=1.5, label='Actual')
    ax.plot(time_axis, predictions[:, i], 'r--', lw=1.5, label='LSTM predicted')
    ax.axvline(time_axis[0], color='gray', linestyle=':', lw=1, label='Prediction start')
    ax.set_title(name)
    ax.set_xlabel('time (s)')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
