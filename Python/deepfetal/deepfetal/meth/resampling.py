# Resampling

from ..build.shift import *
from ..meth.t import *
from ..meth.tmtx import *


def resampling(x, N, fo=0, cosine_domain=False):
    # fo 0: not transformed / fo 1: transformed / fo 2: transformed shifted no renormalization
    NX, N = maketensor(x.shape).type(torch.short), maketensor(N).type(torch.short)
    ND, NDX = len(N) - 1, len(NX) - 1
    is_real = NX[NDX] == 1

    if ND > NDX:  # Size of N larger than size of X
        for n in range(ND-NDX):
            x = torch.unsqueeze(x, -1)
        NX = maketensor(x.shape).type(torch.short)
        NDX = len(NX) - 1
    if NDX > ND:
        N = torch.cat((N, torch.zeros(NDX-ND).type(torch.short)))
        ND = len(N) - 1
    Nmax, Nmin = torch.max(NX, N), torch.min(NX, N)
    nzero = (torch.ceil(torch.true_divide(Nmax+1, 2)) - 1).type(torch.short)
    norig = nzero - (torch.ceil(torch.true_divide(Nmin-1, 2))).type(torch.short)
    nfina = nzero + (torch.floor(torch.true_divide(Nmin-1, 2)) + 1).type(torch.short)
    if cosine_domain:  #  and fo != 2:
        norig = torch.zeros(ND).type(torch.short)
        nfina = Nmin
    for d in range(ND):
        if N[d] != NX[d] and N[d] != 0:
            Nor, Nde = NX[d], N[d]
            NN = NX.clone().detach()
            NN[d] = N[d]
            v = torch.zeros(torch.max(Nor, Nde)).type(torch.bool)
            v[norig[d]:nfina[d]] = True
            if not cosine_domain and fo != 2:
                v = idftshift(v)
            # v = torch.nonzero(v).squeeze(-1)
            if not fo:  # Not in transformed domain
                if cosine_domain:
                    A = matchtypes(dctmtx(Nor), x)
                    AH = matchtypes(dctmtx(Nde), x)
                else:
                    A = matchtypes(dftmtx(Nor), x)
                    AH = matchtypes(dftmtx(Nde), x)
                if Nor > Nde:
                    A = A[v, :]
                if Nor < Nde:
                    AH = AH[v, :]
                A = cmm(ch(AH), A)
                if is_real:
                    A = ccomplex(creal(A))
                xr = apl(A, x, d)
            else:
                if Nor < Nde:
                    xr = matchtypes(torch.zeros(tuple(NN)), x)
                    xr = dynind(xr, v, d, x)
                else:
                    xr = dynind(x, v, d)
            if fo != 2:
                x = xr*torch.sqrt(torch.true_divide(Nde, Nor))
            else:
                x = xr
            NX = maketensor(x.shape).type(torch.short)
    return x
