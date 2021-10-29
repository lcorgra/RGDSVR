# Extensions for matching arrays

import torch

def matchdims(x, y):
    while y.ndim < x.ndim:
        y = torch.unsqueeze(y, -1)
    while x.ndim < y.ndim:
        x = torch.unsqueeze(x, -1)
    return x, y


def matchtypes(x, y):
    return x.type(y.dtype)


def matchviews(x, y):
    x, y = matchdims(x, y)
    NX = torch.tensor(x.shape)
    NY = torch.tensor(y.shape)
    Nex = -1*torch.ones(len(NY)).int().numpy()
    exd = torch.logical_and(NY == 1, NX > 1)
    Nex[exd] = NX[exd]
    y = y.expand(torch.Size(Nex))
    Nex = -1*torch.ones(len(NX)).int().numpy()
    exd = torch.logical_and(NX == 1, NY > 1)
    Nex[exd] = NY[exd]
    x = x.expand(torch.Size(Nex))
    return x, y


def maketensor(x, N=1):  # N is the dimensionality of the tensor
    x = torch.tensor(x)
    while x.ndim < N:
        x = torch.unsqueeze(x, -1)
    return x


def repeattensor(x, N):
    if len(x) == 1:
        return x.repeat(N)
    else:
        return x


