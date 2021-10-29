# Extensions for complex operations

import torch
from ..build.bmul import *
from ..build.dynind import *


def ccomplex(x, y=None):  # Builds complex data from real and imaginary
    if y is None:
        return torch.unsqueeze(x, -1)
    else:
        return torch.stack((x, y), x.ndim)


def ceuler(x): #  Computes the Euler formula
    return ccomplex(torch.cos(x), torch.sin(x))


def creal(x):  # Gets real component from complex data
    ndx = x.ndim - 1
    if x.shape[ndx] == 2:
        return dynind(x, torch.tensor(0), ndx).squeeze(ndx)
    else:
        return x.squeeze(ndx)


def cimag(x):  # Gets imaginary component from complex data
    ndx = x.ndim - 1
    if x.shape[ndx] == 2:
        return dynind(x, torch.tensor(1), ndx).squeeze(ndx)
    else:
        return torch.zeros(x.shape).squeeze(ndx)


def cconj(x):  # Conjugates complex data
    ndx = x.ndim - 1
    if x.shape[ndx] == 2:
        return ccomplex(creal(x), -cimag(x))
    else:
        return x


def cabs(x):  # Complex absolute value
    ndx = x.ndim - 1
    if x.shape[ndx] == 2:
        return torch.sqrt(creal(x) ** 2 + cimag(x) ** 2)
    else:
        return torch.abs(creal(x))


def cangle(x):  # Complex angle
    return torch.atan2(cimag(x), creal(x))


def csign(x):  # Complex sign
    ndx = x.ndim - 1
    if x.shape[ndx] == 2:
        xabs = cabs(x)
        cc = xabs < 1e-12
        x[~cc] = cmul(x[~cc], 1/xabs[~cc])  # Quick route
        x[cc] = ceuler(cangle(x[cc]))  # Slow route
        return x
    else:
        return torch.sign(x)


def cexp(x):  # Complex exponential
    ndx = x.ndim - 1
    if x.shape[ndx] == 2:
        return cmul(torch.exp(creal(x)), ceuler(cimag(x)))
    else:
        return torch.exp(x)


def clog(x):  # Complex logarithm
    return ccomplex(torch.log(cabs(x)), cangle(x))


def ch(x):  # Hermitian matrix
    return cconj(torch.transpose(x, 0, 1))


def cadd(x, y):  # Complex addition
    ndx, ndy = x.ndim-1, y.ndim-1
    if x.shape[ndx] == 2 and y.shape[ndy] == 2:
        return x+y
    elif y.shape[ndy] == 2:
        return ccomplex(creal(x)+creal(y), cimag(y))
    elif x.shape[ndx] == 2:
        return ccomplex(creal(x)+creal(y), cimag(x))
    else:
        return x+y


def csub(x, y):  # Complex subtraction
    return cadd(x, -y)


def cmul(x, y):  # Complex dot multiplication
    ndx, ndy = x.ndim-1, y.ndim-1
    if x.shape[ndx] == 2 and y.shape[ndy] == 2:
        xr, xi, yr, yi = creal(x), cimag(x), creal(y), cimag(y)
        return ccomplex(bmul(xr, yr)-bmul(xi, yi), bmul(xr, yi)+bmul(xi, yr))
    elif y.shape[ndy] == 2:
        xr = creal(x)
        return ccomplex(bmul(xr, creal(y)), bmul(xr, cimag(y)))
    elif x.shape[ndx] == 2:
        yr = creal(y)
        return ccomplex(bmul(creal(x), yr), bmul(cimag(x), yr))
    else:
        return ccomplex(bmul(creal(x), creal(y)))


def cdiv(x, y):  # Complex dot division
    return cmul(x, cmul(ccomplex(1/(cabs(y) **2)), cconj(y)))


def cpow(x, y):  # Complex power, note second is real!
    return cmul(ccomplex(torch.pow(cabs(x), y)), ceuler(torch.mul(cangle(x), y)))


def cmm(x, y):  # Complex 2D matrix multiplication
    ndx, ndy = x.ndim-1, y.ndim-1
    if x.shape[ndx] == 2 and y.shape[ndy] == 2:
        xr, xi, yr, yi = creal(x), cimag(x), creal(y), cimag(y)
        return ccomplex(torch.mm(xr, yr)-torch.mm(xi, yi), torch.mm(xr, yi)+torch.mm(xi, yr))
    elif y.shape[ndy] == 2:
        xr = creal(x)
        return ccomplex(torch.mm(xr, creal(y)), torch.mm(xr, cimag(y)))
    elif x.shape[ndx] == 2:
        yr = creal(y)
        return ccomplex(torch.mm(creal(x), yr), torch.mm(cimag(x), yr))
    else:
        return ccomplex(torch.mm(creal(x), creal(y)))


def cbmm(x, y):  # Complex batched 2D (i.e., 3D) matrix multiplication
    ndx, ndy = x.ndim-1, y.ndim-1
    if x.shape[ndx] == 2 and y.shape[ndy] == 2:
        xr, xi, yr, yi = creal(x), cimag(x), creal(y), cimag(y)
        return ccomplex(torch.bmm(xr, yr)-torch.bmm(xi, yi), torch.bmm(xr, yi)+torch.bmm(xi, yr))
    elif y.shape[ndy] == 2:
        xr = creal(x)
        return ccomplex(torch.bmm(xr, creal(y)), torch.bmm(xr, cimag(y)))
    elif x.shape[ndx] == 2:
        yr = creal(y)
        return ccomplex(torch.bmm(creal(x), yr), torch.bmm(cimag(x), yr))
    else:
        return ccomplex(torch.bmm(creal(x), creal(y)))


def cmatmul(x, y):  # Complex batched and broadcasted matrix multiplication
    ndx, ndy = x.ndim-1, y.ndim-1
    if x.shape[ndx] == 2 and y.shape[ndy] == 2:
        xr, xi, yr, yi = creal(x), cimag(x), creal(y), cimag(y)
        return ccomplex(torch.matmul(xr, yr)-torch.matmul(xi, yi), torch.matmul(xr, yi)+torch.matmul(xi, yr))
    elif y.shape[ndy] == 2:
        xr = creal(x)
        return ccomplex(torch.matmul(xr, creal(y)), torch.matmul(xr, cimag(y)))
    elif x.shape[ndx] == 2:
        yr = creal(y)
        return ccomplex(torch.matmul(creal(x), yr), torch.matmul(cimag(x), yr))
    else:
        return ccomplex(torch.matmul(creal(x), creal(y)))
