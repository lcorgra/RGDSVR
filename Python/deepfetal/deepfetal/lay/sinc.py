import torch
import torch.nn as nn


class SincLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        c = torch.abs(x) > 1e-6
        x[c] = torch.sin(x[c]) / x[c]
        x[~c] = 1
        return x

