import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, config):
    device = config.get('device', 'cpu')
    model  = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()  # regression task — was CrossEntropyLoss (classification only)

    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss   = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{config['num_epochs']} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if 'checkpoint_path' in config:
                save_checkpoint(model, optimizer, epoch, config['checkpoint_path'])


def validate(model, dataloader, criterion, device):
    model.eval()  # disables dropout, uses batchnorm running stats
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:  # was dataloader.val_data — DataLoader is an iterator
            inputs  = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
    model.train()  # reset flag
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.layers import build_rnn

    SEQ_LEN    = 20
    N_FEATURES = 4
    BATCH_SIZE = 32

    # Dummy data — replace with real parquet pipeline when ready
    X = torch.randn(1000, SEQ_LEN, N_FEATURES)
    Y = torch.randn(1000, N_FEATURES)

    split        = int(0.8 * len(X))
    train_loader = DataLoader(TensorDataset(X[:split], Y[:split]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X[split:], Y[split:]), batch_size=BATCH_SIZE)

    # Build a simple model using the layer factory
    rnn  = build_rnn('LSTM', input_size=N_FEATURES, hidden_size=64)
    head = nn.Linear(64, N_FEATURES)
    model = nn.Sequential()  # placeholder — wire rnn + head into a proper nn.Module

    config = {
        'learning_rate':   1e-3,
        'num_epochs':      20,
        'device':          'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_path': 'best_model.pt',
    }

    # train_model(model, train_loader, val_loader, config)
    # Uncomment above once model class is wired up properly
