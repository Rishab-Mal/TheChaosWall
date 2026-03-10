import torch
import torch.nn as nn


class HNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):  # noqa: ARG002 (HNN always outputs scalar H)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def time_derivatives(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().requires_grad_(True)
        H = self.forward(x)
        gradH = torch.autograd.grad(
            H.sum(),
            x,
            create_graph=True,
        )[0]

        # x = [q1, q2, p1, p2]; gradH = [dH/dq1, dH/dq2, dH/dp1, dH/dp2]
        dHdq, dHdp = torch.chunk(gradH, 2, dim=1)

        dqdt = dHdp     # Hamilton: dq/dt =  dH/dp
        dpdt = -dHdq    # Hamilton: dp/dt = -dH/dq
        return torch.cat((dqdt, dpdt), dim=1)

    # Alias used by train.py
    get_derivatives = time_derivatives
