# Vanilla RNN for regression on dummy data

# Model Architecture:
# Input (batch, seq_len, 4)
#         ↓
# Simple RNN (1 layer, 64 hidden units, tanh)
#         ↓
# Take hidden state at final timestep
#         ↓
# Fully Connected Linear Layer (64 → 4)
#         ↓
# Output (batch, 4)

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Reproducibility
torch.manual_seed(67)

# Data
SEQ_LEN    = 20
N_FEATURES = 4
N_SAMPLES  = 1000  # sampled training windows per epoch-length dataset

# Make project/data importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from data.utils import load_sims_from_parquet, sample_subsequence


class ParquetSequenceDataset(Dataset):
    """Samples sequence->next-step pairs from pendulum parquet simulations."""
    def __init__(self, file_path: Path, sim_dict: dict, seq_len: int = 20, n_samples: int = 1000):
        self.file_path = str(file_path)
        self.sim_dict = sim_dict
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.feature_cols = ["theta1", "theta2", "theta1_dot", "theta2_dot"]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Pull seq_len + 1 timesteps so we can predict the next state after the sequence.
        df = sample_subsequence(self.file_path, self.sim_dict, n_samples=self.seq_len + 1)
        features = df.select(self.feature_cols).to_numpy().astype("float32")
        X = torch.tensor(features[:-1], dtype=torch.float32)
        y = torch.tensor(features[-1], dtype=torch.float32)
        return X, y


preferred_path = Path(__file__).resolve().parents[2] / "test_pundulum.parquet"
fallback_path = Path(__file__).resolve().parents[2] / "test_pendulum2.parquet"
if preferred_path.exists():
    parquet_path = preferred_path
elif fallback_path.exists():
    parquet_path = fallback_path
else:
    raise FileNotFoundError(
        f"Could not find parquet dataset. Checked: {preferred_path} and {fallback_path}"
    )

sim_dict = load_sims_from_parquet(str(parquet_path))
dataset = ParquetSequenceDataset(parquet_path, sim_dict, seq_len=SEQ_LEN, n_samples=N_SAMPLES)
train_split = int(0.8 * len(dataset))
train_dataset = Subset(dataset, range(train_split))
test_dataset = Subset(dataset, range(train_split, len(dataset)))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32)

# Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc  = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])

model = RNNModel(input_size=N_FEATURES, hidden_size=64, output_size=N_FEATURES).to(device)

# Loss and optimizer
loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Training loop
torch.manual_seed(67)
epochs = 100

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch.to(device))
        loss   = loss_fn(y_pred, y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Eval
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            test_pred  = model(X_batch.to(device))
            test_loss += loss_fn(test_pred, y_batch.to(device)).item()

    print(f"Epoch: {epoch+1:3d} | Loss: {train_loss/len(train_loader):.4f} | Test loss: {test_loss/len(test_loader):.4f}")

# Save model
MODEL_PATH      = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_PATH / "rnn.pth"
torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
print(f"Model saved to: {MODEL_SAVE_PATH}")

# Load model
loaded_model = RNNModel(input_size=N_FEATURES, hidden_size=64, output_size=N_FEATURES)
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, weights_only=True))
loaded_model.to(device)


