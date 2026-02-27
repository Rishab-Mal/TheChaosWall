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
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
SEQ_LEN    = 20
N_FEATURES = 4
N_SAMPLES  = 1000

X = torch.randn(N_SAMPLES, SEQ_LEN, N_FEATURES)
y = torch.randn(N_SAMPLES, N_FEATURES)

# Split data
train_split = int(0.8 * len(X))
X_train, X_test = X[:train_split], X[train_split:]
y_train, y_test = y[:train_split], y[train_split:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=32)

# Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc  = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])

torch.manual_seed(67)
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
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model.to(device)

# Plot predictions (first feature across test samples)
model.eval()
with torch.inference_mode():
    y_preds = model(X_test.to(device)).cpu()

plt.figure(figsize=(10, 7))
plt.plot(y_test[:, 0],  c="g", label="True")
plt.plot(y_preds[:, 0], c="r", label="Predicted")
plt.legend()
plt.show()