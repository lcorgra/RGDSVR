import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
