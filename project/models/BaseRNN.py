import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset  # WRONG: DataLoader/TensorDataset belong in training code, not a model file

# WRONG: build_rnn here duplicates stacked_rnn_layers() in layers.py — two functions doing the same job.
#        Pick one place and delete the other. layers.py version is more complete (supports plain 'RNN' type).
# WRONG: args are passed positionally to torch.nn.LSTM/GRU — if PyTorch ever reorders them this silently breaks.
#        Use keyword arguments: nn.LSTM(input_size=..., hidden_size=..., ...)
# WRONG: plain 'RNN' type is not handled here, only LSTM/GRU — layers.py handles all three
def build_rnn(rnn_type, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False):
    if rnn_type == 'LSTM':
        return torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
    elif rnn_type == 'GRU':
        return torch.nn.GRU(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
    else:
        raise ValueError("Unsupported RNN type: {}".format(rnn_type))

class BaseRNN(torch.nn.Module):
    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0,
        bidirectional=False
        # NOT DONE: output_size is missing as a parameter — the __main__ block below passes it
        #           but __init__ doesn't accept it, so instantiation will crash with TypeError.
        #           Add output_size here and create a self.fc = nn.Linear(hidden_size, output_size)
    ):
        super(BaseRNN, self).__init__()

        self.rnn = build_rnn(rnn_type, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
        # NOT DONE: no output projection layer — the RNN outputs shape (batch, seq, hidden_size)
        #           but we want to predict shape (batch, output_size). Need:
        #           self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        # WRONG: self.rnn(x, h) returns a tuple (output, hidden_state) — not a plain tensor.
        #        The training loop does criterion(pred, Y_batch) where pred = model(X_batch),
        #        so pred is a tuple and MSELoss will crash. Need to unpack: output, _ = self.rnn(x, h)
        #        Then apply self.fc to the last timestep: return self.fc(output[:, -1, :])
        return self.rnn(x, h)

    def init_hidden(self, batch_size):
        # GOOD: correctly handles LSTM (needs h_0, c_0 tuple) vs GRU/RNN (just h_0)
        num_directions = 2 if self.rnn.bidirectional else 1
        h_0 = torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size)
        # WRONG: h_0 and c_0 are always on CPU — if model is on GPU this will cause a device mismatch.
        #        Should be: torch.zeros(..., device=next(self.parameters()).device)
        if isinstance(self.rnn, torch.nn.LSTM):
            c_0 = torch.zeros(self.rnn.num_layers * num_directions, batch_size, self.rnn.hidden_size)
            return (h_0, c_0)
        else:
            return h_0

    def stacked_rnn_layers(self, x):
        # WRONG: this method name is misleading — it doesn't stack anything, it just runs a forward pass.
        #        It also duplicates what forward() does. Either remove this method or rename it.
        output, _ = self.rnn(x)
        return output

# ── data (replace with your real data) ─────────────────────────────
# WRONG: make_dummy_data, train(), and the __main__ block all belong in training/train.py, not the model file.
#        Model files should only define model architecture. Having training code here makes it impossible
#        to import BaseRNN without also importing DataLoader etc.
def make_dummy_data(n_samples=1000, seq_len=20, n_features=4):
    """
    Replace this with your actual physics data.
    X : (n_samples, seq_len, n_features)  — past 20 states
    Y : (n_samples, n_features)           — next state to predict
    """
    X = torch.randn(n_samples, seq_len, n_features)
    Y = torch.randn(n_samples, n_features)
    return X, Y


# ── training + evaluation ───────────────────────────────────────────
# WRONG: this train() function should be in training/train.py — move it there
def train(model, train_loader, val_loader, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):

        # ── train ──────────────────────────────────────────────────
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)           # forward
            # WRONG: pred is a tuple (output, hidden) from BaseRNN.forward() — MSELoss will crash here
            loss = criterion(pred, Y_batch) # compare to true next state
            loss.backward()                 # backprop
            optimizer.step()                # update weights
            train_loss += loss.item()

        # ── validate ───────────────────────────────────────────────
        # GOOD: model.eval() + torch.no_grad() is the correct validation pattern
        model.eval()
        val_loss = 0
        with torch.no_grad():               # no gradients needed
            for X_batch, Y_batch in val_loader:
                pred = model(X_batch)
                # WRONG: same issue — pred is a tuple
                val_loss += criterion(pred, Y_batch).item()

        # GOOD: averaging loss over batches by dividing by len(loader) is correct
        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss:   {val_loss/len(val_loader):.4f}")
        # NOT DONE: no learning rate scheduler step here
        # NOT DONE: no checkpoint saving here
        # NOT DONE: no early stopping


# ── run ─────────────────────────────────────────────────────────────
# WRONG: this __main__ block should be in training/train.py, not the model file
if __name__ == "__main__":
    SEQ_LEN    = 20
    N_FEATURES = 4
    BATCH_SIZE = 32

    # load your data here
    X, Y = make_dummy_data(n_samples=1000, seq_len=SEQ_LEN, n_features=N_FEATURES)

    # split 80/20 train/test
    split     = int(0.8 * len(X))
    train_ds  = TensorDataset(X[:split], Y[:split])
    val_ds    = TensorDataset(X[split:], Y[split:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # WRONG: output_size=N_FEATURES is passed here but BaseRNN.__init__ doesn't accept output_size — TypeError crash
    # WRONG: rnn_type is also not passed — will crash on missing required positional argument
    model = BaseRNN(input_size=N_FEATURES, hidden_size=64, output_size=N_FEATURES)

    train(model, train_loader, val_loader, epochs=20)
