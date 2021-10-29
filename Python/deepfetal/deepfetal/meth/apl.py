# Apply matrix A over m dimension of array x

import torch
import numpy as np
from deepfetal.build.complex import *


def apl(A, x, m):
    N = x.shape
    ND = len(N) - 1
    N = np.array(N)
    M = N[:ND]
    if m > ND-1:
        raise SyntaxError('Dimension of application larger than dimension of array')
    elif m == 0:
        x = torch.reshape(x, (M[0], np.prod(M[1:]), N[ND]))
        x = cmm(A, x)
        NDX = 2
    elif m != ND - 1:
        x = torch.reshape(x, (np.prod(M[:m]), M[m], np.prod(M[m+1:]), N[ND]))
        x = cmatmul(A, x)
        NDX = 3
    else:
        x = torch.reshape(x, (np.prod(M[:m]), M[m], N[ND]))
        x = cmm(x, torch.transpose(A, 0, 1))
        NDX = 2
    N[m] = A.shape[0]
    N[ND] = x.shape[NDX]
    x = torch.reshape(x, list(N))
    return x
