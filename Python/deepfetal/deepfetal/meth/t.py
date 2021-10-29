# Discrete {Cosine,Fourier} Transforms along given dimensions

from deepfetal.build.complex import *
from ..build.matcharrays import *
from ..meth.apl import *
from ..meth.tmtx import *


def dct(x, d=0, A=None):
    N = x.shape[d]
    if A is None:
        A = matchtypes(dctmtx(N), x)
    return apl(A, x, d)


def idct(x, d=0, AH=None):
    N = x.shape[d]
    if AH is None:
        AH = matchtypes(idctmtx(N), x)
    return apl(AH, x, d)


def dft(x, d=0, A=None):
    N = x.shape[d]
    if A is None:
        A = matchtypes(dftmtx(N), x)
    return apl(A, x, d)


def idft(x, d=0, AH=None):
    N = x.shape[d]
    if AH is None:
        AH = matchtypes(idftmtx(N), x)
    return apl(AH, x, d)


def dwt(x, d=0, A=None, typ='db1', L=1, over=False):
    N = x.shape[d]
    if A is None:
        A = matchtypes(dwtmtx(N, typ=typ, L=L, over=over), x)
    return apl(A, x, d)


def idwt(x, d=0, A=None, typ='db1', L=1):
    N = x.shape[d]
    if A is None:
        A = matchtypes(idwtmtx(N, typ=typ, L=L, over=over), x)
    return apl(A, x, d)
