import torch.nn as nn
from torch import Tensor
import numpy as np
from ..lay.encode import *


def lossTodB(x):
    return -10.*np.log10(x.data.cpu().numpy())


class EncodeMSELoss(nn.MSELoss):
    __constants__ = ['reduction']

    def __init__(self, encodingFunc=encoderUnit, encodingStruct=None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(EncodeMSELoss, self).__init__(size_average, reduce, reduction)
        self.encodingFunc, self.encodingStruct = encodingFunc, encodingStruct

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return nn.functional.mse_loss(self.encodingFunc(input, self.encodingStruct), target, reduction=self.reduction)
