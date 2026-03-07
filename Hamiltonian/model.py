import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import config 
class HNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(config.HIDDEN_DIM, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def time_derivatives(self, x: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        H = self.forward(x)
       
        gradH = torch.autograd.grad(
           H.sum(),
           x, 
           create_graph=True
        )[0]

        q, p = torch.chunk(x, 2, dim=1)
        dHdp, dHdq = torch.chunk(gradH, 2, dim=1)

        dqdt = dHdp
        dpdt = -dHdq
        return torch.cat((dqdt, dpdt), dim=1)
    
