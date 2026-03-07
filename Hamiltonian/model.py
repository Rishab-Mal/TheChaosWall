import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class HNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh(),
            nn.Linear(output_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)