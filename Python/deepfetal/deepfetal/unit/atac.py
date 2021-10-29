import torch
import torch.nn as nn
import numpy as np
from ..lay.sinc import SincLayer


class AtacUnit(nn.Module):
    def __init__(self, Kin, Kred=1, ND=2, use_bias=False, use_bn=1, use_sine=False):
        # PARAMETERS
        super().__init__()
        Kou = np.maximum(np.ceil(Kin / Kred), 1).astype(int)

        # LAYERS
        if use_sine:
            self.convolver1 = nn.Linear(Kin, Kou, bias=use_bias)
            self.batchnorm1 = nn.BatchNorm1d(Kou)
            self.convolver2 = nn.Linear(Kou, Kin, bias=use_bias)
            self.batchnorm2 = nn.BatchNorm1d(Kin)
        else:
            if ND == 2:
                self.convolver1 = nn.Conv2d(Kin, Kou, 1, 1, padding=0, bias=use_bias)
                self.batchnorm1 = nn.BatchNorm2d(Kou)
                self.convolver2 = nn.Conv2d(Kou, Kin, 1, 1, padding=0, bias=use_bias)
                self.batchnorm2 = nn.BatchNorm2d(Kin)
            else:
                self.convolver1 = nn.Conv3d(Kin, Kou, 1, 1, padding=0, bias=use_bias)
                self.batchnorm1 = nn.BatchNorm3d(Kou)
                self.convolver2 = nn.Conv3d(Kou, Kin, 1, 1, padding=0, bias=use_bias)
                self.batchnorm2 = nn.BatchNorm3d(Kin)
        if use_bn < 0:
            self.batchnorm1 = nn.Identity()
        if use_bn < 1:
            self.batchnorm2 = nn.Identity()

        self.activation1 = nn.ReLU()
        if use_sine:
            self.activation2 = SincLayer()
        else:
            self.activation2 = nn.Sigmoid()

    def forward(self, x):
        return x*self.activation2(self.batchnorm2(self.convolver2(self.activation1(self.batchnorm1(self.convolver1(x))))))
