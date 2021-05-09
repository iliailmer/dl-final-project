import torch
from torch import nn


class ReSU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x, x.sigmoid())
