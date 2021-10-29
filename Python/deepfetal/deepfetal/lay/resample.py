import torch.nn as nn
from ..meth.resampling import *
from ..meth.tmtx import *


class ResampleLayer(nn.Module):
    def __init__(self, NInput, NOutput, use_dct=True):
        super().__init__()
        NInput = np.atleast_1d(np.array(NInput))
        NOutput = np.atleast_1d(np.array(NOutput))
        self.ND = NInput.size
        self.NInput, self.NOutput = NInput, NOutput
        NMin = np.minimum(NInput, NOutput)
        for n in range(self.ND):
            if use_dct:  # COSINE
                Din = dctmtx(NInput[n])
                Dou = dctmtx(NOutput[n])
                Din = Din[range(int(NMin[n])), :, :]
                Dou = Dou[range(int(NMin[n])), :, :]
            else:  # FOURIER
                Din = dftmtx(NInput[n])
                Dou = dftmtx(NOutput[n])
                Din = resampling(dftshift(Din, 0), NMin[n], 2)
                Dou = resampling(dftshift(Dou, 0), NMin[n], 2)
            F = ccomplex(creal(cmm(ch(Dou), Din)))  # We assume it is real
            if n == 0:
                self.register_buffer('F1', F)
            elif n == 1:
                self.register_buffer('F2', F)
            elif n == 2:
                self.register_buffer('F3', F)

    def forward(self, x):
        for n in range(self.ND):
            if n == 0:
                x = creal(apl(self.F1, ccomplex(x), n+2))
            elif n == 1:
                x = creal(apl(self.F2, ccomplex(x), n+2))
            elif n == 2:
                x = creal(apl(self.F3, ccomplex(x), n+2))
        return x
