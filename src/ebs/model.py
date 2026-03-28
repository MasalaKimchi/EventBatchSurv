from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


class CoxMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2):
        super().__init__()
        dims = [input_dim, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CoxResNet18(nn.Module):
    """ResNet18 Cox backbone matching TorchSurv tutorial defaults."""

    def __init__(self):
        super().__init__()
        net = resnet18(weights=None)
        # Tutorial-style grayscale stem and single log-risk head.
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=1)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
