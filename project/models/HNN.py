# TODO: description

import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable
#from torch.utils.data import DataLoader, TensorDataset
#import matplotlib.pyplot as plt
#from pathlib import Path

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

class HNNModel(nn.Module):
    def __init__(self, hamiltonian_fn: Callable[...], input_size: int, latent_depth: int = 3, latent_size: int = 200):
        super().__init__()
        self.latent_size = latent_size
        self.hamiltonian_fn = hamiltonian_fn
        self.input_size = input_size

        self.latent_fc = [
            nn.Linear(in_features=input_size, out_features=latent_size) if i == 0 
            else nn.Linear(in_features=latent_size, out_features=latent_size) for i in range(latent_depth)
        ]
        self.hamiltonian_fc = nn.Linear(in_features=latent_size, out_features=1)

    def hamiltonian_pred(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != device: x = x.to(device)
        for l in self.latent_fc:
            x = F.tanh(l(x))
        hamiltonian = F.tanh(self.hamiltonian_fc(x))
        return hamiltonian
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad = torch.autograd.grad(self.hamiltonian_pred(x).sum(), x, create_graph=True)[0]


        pass # TODO
        #...


# Adam optimizer, 10e-3 learning rate, 2000 gradient steps, tanh activations, 200 hidden units
# "We logged three metrics: L2 train loss, L2 test loss, and mean squared error (MSE) between the true
# and predicted total energies"