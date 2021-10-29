import torch
import torch.nn as nn
import numpy as np
from ..lay.tanh import TanhLayer
from ..lay.sine import SineLayer
from ..lay.swish import SwishLayer
from ..lay.resample import ResampleLayer
from ..unit.atac import AtacUnit


class DecoderUnit(nn.Module):
    def __init__(self, Kin, Kou, id, Nin, Nou, NCo=1, ND=2, use_bias=False, use_act=0, use_conv_bn=0, use_sinc=0,
                 upsample_first=True, Kred=1, use_bias_atac=False, use_dct=True):
        # PARAMETERS
        super().__init__()
        pi, stride, to_pad, self.upsample_first = np.pi, 1, int((NCo - 1) / 2), upsample_first
        self.use_conv_bn = use_conv_bn

        # LAYERS
        if ND == 2:
            self.padder = nn.ReflectionPad2d(to_pad)
            self.convolver = nn.Conv2d(Kin, Kou, NCo, stride, padding=to_pad, bias=use_bias)
            #  self.batchnorm = nn.BatchNorm2d(Kou, track_running_stats=False)
            self.batchnorm = nn.BatchNorm2d(Kou)
        else:
            #self.padder = nn.ReflectionPad3d(to_pad)
            self.padder = nn.Identity()  # There is no built-in padder in 3D
            self.convolver = nn.Conv3d(Kin, Kou, NCo, stride, padding=to_pad, bias=use_bias)
            #  self.batchnorm = nn.BatchNorm3d(Kou, track_running_stats=False)
            self.batchnorm = nn.BatchNorm3d(Kou)
        if to_pad == 0:
            self.padder = nn.Identity()

        if use_conv_bn == -1:
            self.batchnorm = nn.Identity()

        if use_sinc == 0:
            self.resampler = nn.Upsample(scale_factor=int(Nou[0]/Nin[0]), mode='bilinear')
        elif use_sinc == -1:
            self.resampler = nn.Upsample(scale_factor=int(Nou[0]/Nin[0]), mode='nearest')
        else:
            self.resampler = ResampleLayer(Nin, Nou, use_dct=use_dct)

        self.activation = nn.Identity()
        #
        if use_act == 0:
            self.activation = nn.ReLU()
        elif use_act == 1:
            self.activation = SwishLayer()
        elif use_act == 2:
            self.activation = SineLayer()
        elif use_act == 3:
            self.activation = TanhLayer()
        elif use_act == 4:
            self.activation = AtacUnit(Kou, Kred=Kred, ND=ND, use_bias=use_bias_atac)
        if use_act == 5:
            self.batchnorm = nn.Identity()
            self.resampler = nn.Identity()

        # INITIALIZE
        with torch.no_grad():
            # if use_act == 5:  # No activation
            #    nn.init.normal_(self.convolver.weight, 0., np.sqrt(1. / (2. * Kin)))
            #    if use_bias:
            #        #nn.init.uniform_(self.convolver.bias, 0, 0)
            #        nn.init.uniform_(self.convolver.bias, -2 * pi, -2 * pi)
            if use_act == 2:  # Sine
                nn.init.uniform_(self.convolver.weight, -np.sqrt(6. / Kin), np.sqrt(6. / Kin))
                if use_bias:
                    nn.init.uniform_(self.convolver.bias, -pi, pi)
            #else:
            #    if use_bias:
            #        nn.init.uniform_(self.convolver.bias, 0, 0)

    def forward(self, x):
        if self.upsample_first:
            x = self.resampler(self.convolver(x))
        else:
            x = self.convolver(self.resampler(x))
        if self.use_conv_bn == 1:
            return self.activation(self.batchnorm(x))
        else:
            return self.batchnorm(self.activation(x))
