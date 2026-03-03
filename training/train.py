import torch
from torch.utils.data import DataLoader


def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for i, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss



def train_model(config):
    # Load the dataset
    train_data, val_data = load_dataset(config['dataset_path'])
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize the model, optimizer, and loss function
    model = MyModel(config['model_params'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(config['num_epochs']):
        train_one_epoch(model, train_loader, optimizer, epoch)
        validate(model, val_loader, criterion, config['device'])







def validate(model, dataloader, criterion, device):
    model.eval()  # flag, necesary to have model behave diferent in validation than training
    with torch.no_grad():
        inputs = dataloader.val_data.to(device)    #get input data from prev motion
        targets = dataloader.val_targets.to(device) # shifted by 1 timestep
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    model.train() # reset flag
    return loss

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)                          # load the dict you saved
    model.load_state_dict(checkpoint['model_state_dict'])  # restore weights
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # restore optimizer
    epoch = checkpoint['epoch']                            # where you left off
    return model, optimizer, epoch