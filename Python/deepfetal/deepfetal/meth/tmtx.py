# Discrete {Cosine,Fourier} Transform matrix of N elements

import torch
import numpy as np
import pywt
from deepfetal.build.complex import *
from deepfetal.build.dynind import *


def dctmtx(N, inv=False):
    N = float(N)
    rr, cc = torch.meshgrid(
        [torch.arange(0, N), torch.arange(0, N)])  # This is opposite to matlab meshgrid, i.e., ndgrid probably
    c = np.sqrt(2./N)*torch.cos(np.pi*(2.*cc+1)*rr/(2.*N))
    c[0, :] = c[0, :] / np.sqrt(2.)
    c = ccomplex(c)
    if inv:
        c = ch(c)
    return c


def idctmtx(N):
    return dctmtx(N, inv=True)


def dftmtx(N, inv=False):
    N = float(N)
    rr, cc = torch.meshgrid(
        [torch.arange(0, N), torch.arange(0, N)])  # This is opposite to matlab meshgrid, i.e., ndgrid probably
    rr = cc*((2.*np.pi/N)*rr)
    f = ceuler(-rr)/np.sqrt(N)
    if inv:
        f = ch(f)
    return f


def idftmtx(N):
    return dftmtx(N, inv=True)

def dwtmtx(N, typ='db1', L=1, inv=False, over=False):
    N, L = int(N), int(L)
    w = pywt.Wavelet(typ)  #Wavelet object
    #  w.filter_bank == (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi)
    ha, hd = torch.tensor(w.dec_lo), torch.tensor(w.dec_hi)
    H = torch.stack((ha, hd), 1)
    H = torch.unsqueeze(H, -1)
    H = torch.transpose(H, 0, 2)
    W = torch.eye(N)
    for l in range(L):
        T = torch.eye(0)
        if not over:
            A = waveletmatrix(int(N/(2**l)), H, over)
        else:
            A = waveletmatrix(int(N), H, over)
        for s in range((2**l)):
            T = torch.block_diag(T, A)
        W = torch.mm(T, W)
    W = ccomplex(W)
    if inv:
        W = ch(W)
    return W


def idwtmtx(N, typ='db1', L=1, over=False):
    return dwtmtx(N, typ=typ, L=L, over=over, inv=True)


def waveletmatrix(N, H, over):
    H = torch.flip(H, [2,])
    Z = torch.zeros((1, 2, N - H.shape[2])).type(H.dtype)
    W = torch.cat((H, Z), 2)
    if not over:
        for n in range(int(N/2)-1):
            W = torch.cat((W, torch.roll(dynind(W, n, 0), 2, 2)), 0)
    else:
        W = torch.roll(W, -np.floor((H.shape[2]-1)/2).astype(int), 2)
        for n in range(int(N)-1):
            W = torch.cat((W, torch.roll(dynind(W, n, 0), 1, 2)), 0)
    W = torch.transpose(W, 1, 0)
    if not over:
        return torch.reshape(W, (N, N))
    else:
        return torch.reshape(W, (2*N, N))/np.sqrt(2.)

# def build_grid(N):
#    # Generates a flattened grid of coordinates from -1 to 1
#    ND = len(N)
#    tensors = []
#    for n in range(ND):
#        tensors = tensors + tuple(torch.linspace(-1, 1, steps=N[n]))
#    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
#    mgrid = mgrid.reshape(-1, dim)
#    return mgrid
