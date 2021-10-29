import torch.nn as nn
import numpy as np
from ..unit.decoder import DecoderUnit


class DeepDecoderStruct:
    def __init__(self, deepdecoder_pars):
        # DEFAULTS
        # Filter sizes
        self.K = [128] * 5 + [1]
        # Resolution levels / Output size / Number of dimensions
        self.L, self.N, self.ND = len(self.K)-3, [512] * 2, 2
        # Kernel size / Upsampling factor / Upsample first
        self.NCo, self.NRe, self.upsample_first = None, None, True
        # Use sinc upsampling / Type of activation (0 relu, 1 swish, 2 sine, 3 tanh, 4 none)
        self.use_sinc, self.use_act = 0, 0
        # Use conventional batch normalization order (0 no, 1 yes, -1 for not to use) / Use bias
        self.use_conv_bn, self.use_bias = 0, False
        # Derivative with respect to input / Channel reduction for atac unit / Use bias for atac unit
        self.der_input, self.Kred, self.use_bias_atac = False, 1, False
        # Patch size / Use DCT
        self.P, self.use_dct = self.N, True

        # ASSIGNMENTS
        if not isinstance(deepdecoder_pars, dict):
            deepdecoder_pars = deepdecoder_pars.__dict__
        for x in dir(self):
            if any(y == x for y in deepdecoder_pars):
                setattr(self, x, deepdecoder_pars[x])

        # TYPE CONVERSIONS
        self.K = np.rint(np.array(self.K)).astype(int)  # np.around could replace np.rint but it is more general
        self.L = np.rint(self.L).astype(int)            # (round to any decimal) and expensive

        # NEW DEFAULTS
        if self.NCo is None:
            self.NCo = [1] * (self.L + 2)
        if self.NRe is None:
            self.NRe = [2] * self.L + [1] * 2

        # NEW TYPE CONVERSIONS
        self.NCo = np.rint(np.array(self.NCo)).astype(int)

        # print(self.__dict__)


class DeepDecoderNetwork(nn.Module):
    def __init__(self, deep_decoder_struct=None):
        super().__init__()
        dds = DeepDecoderStruct(deep_decoder_struct)
        self.net, self.der_input = [], dds.der_input

        for i in range(dds.L+2):
            use_act = dds.use_act
            if i == dds.L+1:  # Linear
                use_act = 5
            nin = np.ceil(dds.P / np.prod(dds.NRe[i:]))  # Upsample then convolve
            nou = np.ceil(dds.P / np.prod(dds.NRe[i + 1:]))
            self.net.append(DecoderUnit(dds.K[i], dds.K[i+1], i, nin, nou, NCo=dds.NCo[i], ND=dds.ND,
                                        use_bias=dds.use_bias, use_act=use_act, use_conv_bn=dds.use_conv_bn,
                                        use_sinc=dds.use_sinc, upsample_first=dds.upsample_first,
                                        Kred=dds.Kred, use_bias_atac=dds.use_bias_atac, use_dct=dds.use_dct))
        self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        if self.der_input:
            x = x.clone().detach().requires_grad_(True)  # Allows to take derivative w.r.t. input
        return self.net(x)
