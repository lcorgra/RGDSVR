# fftshift and ifftshift methods

from ..build.dynind import *
from ..build.matcharrays import *


def dftshift(x, dim=0, inv=False):
    dim = maketensor(dim).type(torch.int)
    ND = len(dim)
    N = dynind(maketensor(x.shape), dim).type(torch.float)
    s = torch.floor(N/2.).type(torch.int)
    dim = tuple(dim.numpy())
    if inv:
        s = -s
    s = tuple(s.numpy())
    return torch.roll(x, s, dim)


def idftshift(x, dim=0):
    return dftshift(x, dim, inv=True)
