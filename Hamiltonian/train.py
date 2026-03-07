import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import HamiltonianDynamics
from rnn_model import RNNModel 

def train_hnn(config):
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    rnn = RNNModel(input_size=4, hidden_size=64, output_size=4).to(device)
    rnn.load_state_dict(torch.load(config['rnn_path'], map_location=device))
    rnn.eval() # Usually we freeze the RNN to let the HNN learn the energy manifold
    
    # 2. Initialize HNN
    hnn = HamiltonianDynamics(input_dim=4, hidden_dim=128).to(device)
    optimizer = optim.Adam(hnn.parameters(), lr=config['lr'])
    mse_loss = torch.nn.MSELoss()

    # 3. Load Data (Assumes your data.py provides this)
    # train_loader should yield (sequence_window, target_derivatives)
    train_loader = config['train_loader']

    for epoch in range(config['epochs']):
        hnn.train()
        total_loss = 0

        for batch_idx, (seq_in, target_derivs) in enumerate(train_loader):
            seq_in, target_derivs = seq_in.to(device), target_derivs.to(device)
            with torch.no_grad():
                current_state = rnn(seq_in).detach()
            pred_derivs = hnn.get_derivatives(current_state)
            
            # Loss: How well does HNN's physics match the actual motion? thats because it is learning the energy manifold, not the trajectory
            loss = mse_loss(pred_derivs, target_derivs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['epochs']} | HNN Physics Loss: {avg_loss:.6f}")
        if (epoch + 1) % 10 == 0:
            torch.save(hnn.state_dict(), f"models/hnn_epoch_{epoch+1}.pth")

    return hnn