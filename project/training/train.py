import torch
from torch.utils.data import DataLoader
# NOT DONE: missing imports — need to import BaseRNN (or whatever model class is used),
#           load_sims_from_parquet / Pipeline from data/utils.py, and any loss functions

# WRONG: parameter is named 'num_epochs' but it receives the current epoch index (a single int).
#        Rename to 'epoch' to match how it's called: train_one_epoch(..., epoch)
# WRONG: 'criterion' is used inside but is not a parameter and not defined in scope — NameError at runtime.
#        Add criterion as a parameter: def train_one_epoch(model, dataloader, optimizer, epoch, criterion):
def train_one_epoch(model, dataloader, optimizer, num_epochs):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # WRONG: nn.Module has no .device attribute — this will raise AttributeError.
        #        Fix: pass device as a parameter, or use next(model.parameters()).device
        inputs, targets = inputs.to(model.device), targets.to(model.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        # WRONG: criterion is not defined in this scope — will raise NameError. Add it as a parameter.
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            # WRONG: num_epochs prints the epoch index, not the total — misleading label in the log
            print(f"Epoch [{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")


def train_model(config):
    # WRONG: load_dataset is not defined or imported anywhere — will raise NameError at runtime.
    #        Should call load_sims_from_parquet() from data/utils.py, or the Pipeline class once built.
    train_data, val_data = load_dataset(config['dataset_path'])

    # GOOD: creating separate train/val DataLoaders with shuffle=True only for training is correct
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

    # WRONG: MyModel is not defined or imported — will raise NameError.
    #        Replace with: model = BaseRNN(rnn_type=..., input_size=..., hidden_size=..., output_size=...)
    model = MyModel(config['model_params'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    # WRONG: CrossEntropyLoss is for classification tasks (outputs class probabilities).
    #        For physics state prediction (regression) this should be nn.MSELoss() or nn.L1Loss().
    criterion = torch.nn.CrossEntropyLoss()

    # NOT DONE: model is never moved to the target device — should add: model = model.to(config['device'])
    # NOT DONE: no learning rate scheduler (e.g. ReduceLROnPlateau or CosineAnnealingLR)

    # GOOD: looping over epochs and calling separate train/validate functions is clean structure
    for epoch in range(config['num_epochs']):
        # WRONG: train_one_epoch expects 4 args but criterion is missing — should pass criterion here too
        train_one_epoch(model, train_loader, optimizer, epoch)
        validate(model, val_loader, criterion, config['device'])
        # NOT DONE: no checkpoint saving at end of each epoch or on best val loss
        # NOT DONE: no early stopping logic


def validate(model, dataloader, criterion, device):
    # GOOD: model.eval() is correct to disable dropout and use running stats in batchnorm
    model.eval()  # flag, necesary to have model behave diferent in validation than training
    with torch.no_grad():
        # WRONG: DataLoader has no .val_data or .val_targets attributes — these don't exist.
        #        DataLoader is an iterator. Should loop over it like in train_one_epoch:
        #        for inputs, targets in dataloader: ...
        #        The current code will raise AttributeError immediately.
        inputs = dataloader.val_data.to(device)    #get input data from prev motion
        targets = dataloader.val_targets.to(device) # shifted by 1 timestep
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # NOT DONE: if the above were fixed to loop over batches, you'd need to accumulate and average loss
    # GOOD: resetting model.train() after validation is correct practice
    model.train() # reset flag
    return loss

# GOOD: save_checkpoint saves everything needed to resume training (epoch, weights, optimizer state)
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    # NOT DONE: should also save loss/metrics so you know which checkpoint was best

# GOOD: load_checkpoint correctly restores model weights, optimizer state, and epoch number
def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)                          # load the dict you saved
    model.load_state_dict(checkpoint['model_state_dict'])  # restore weights
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # restore optimizer
    epoch = checkpoint['epoch']                            # where you left off
    return model, optimizer, epoch

# NOT DONE: no __main__ block or config dict to actually run training end-to-end
# NOT DONE: no connection to the data pipeline in data/utils.py
# NOT DONE: no logging (tensorboard, wandb, or even a simple log file)
# NOT DONE: no metric tracking beyond loss (e.g. per-dimension error, trajectory rollout error)
