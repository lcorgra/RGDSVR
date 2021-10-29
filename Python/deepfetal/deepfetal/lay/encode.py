from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from ..build.complex import *
from ..meth.apl import *


class EncodingStruct:
    def __init__(self, encoding_pars):
        # DEFAULTS
        # Sampling mask
        self.A, self.C1, self.C2, self.C3, self.ND = None, None, None, None, None

        if torch.cuda.device_count() > 0:
            type = torch.cuda.FloatTensor
            # type = torch.cuda.HalfTensor
            # type = torch.cuda.BFloat16Tensor
        else:
            type = torch.FloatTensor
            # type = torch.HalfTensor
            # type = torch.BFloat16Tensor

        # ASSIGNMENTS
        if not isinstance(encoding_pars, dict):
            encoding_pars = encoding_pars.__dict__
        for x in dir(self):
            if any(y == x for y in encoding_pars):
                setattr(self, x, encoding_pars[x])

        if self.A is not None:
            self.A = Variable(torch.from_numpy(np.array(self.A))).type(type)
        if self.C1 is not None:
            self.C1 = Variable(torch.from_numpy(np.array(self.C1))).type(type)
        if self.C2 is not None:
            self.C2 = Variable(torch.from_numpy(np.array(self.C2))).type(type)
        if self.C3 is not None:
            self.C3 = Variable(torch.from_numpy(np.array(self.C3))).type(type)
        if self.ND is not None:
            self.ND = np.rint(np.array(self.ND)).astype(int)

        # print(self.__dict__)


def encoderUnit(x, E):
    return x


def encoderTimes(x, E):
    return torch.mul(x, E.A)  # This is same as x * E.A


def encoderCosine(x, E):
    for n in range(E.ND):
        if n == 0 and E.C1 is not None:
            x = creal(apl(E.C1, ccomplex(x), n + 2))
        elif n == 1 and E.C2 is not None:
            x = creal(apl(E.C2, ccomplex(x), n + 2))
        elif n == 2 and E.C3 is not None:
            x = creal(apl(E.C3, ccomplex(x), n + 2))
    return torch.mul(x, E.A)  # This is same as x * E.A


def buildEncoding(typ_enc):
    if typ_enc == 'unit':
        encodeFunc = encoderUnit
    elif typ_enc == 'volumetric':
        encodeFunc = encoderTimes
    elif typ_enc == 'cosine':
        encodeFunc = encoderCosine
    return encodeFunc


class Encode(nn.Module):
    def __init__(self, encodingFunc, encodingStruct):
        super().__init__()
        self.encodingFunc, self.encodingStruct = encodingFunc, encodingStruct

    def forward(self, x):
        return self.encodingFunc(x, self.encodingStruct)
