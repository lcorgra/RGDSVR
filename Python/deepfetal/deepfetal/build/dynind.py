# Dynamic indexing

import torch


def dynind(x, v, d=0, y=None):  # Dynamic indexing
    v = torch.tensor(v)
    if v.dtype == torch.bool:
        v = torch.nonzero(v).squeeze(-1)
    if v.dtype is not torch.int64:
        v = v.type(torch.int64)
    if y is None:
        return torch.index_select(x, d, v)
    else:
        return x.index_copy_(d, v, y)


def dynindadd(x, v, d, y):  # Dynamic indexing addition
    v = torch.tensor(v)
    if v.dtype == torch.bool:
        v = torch.nonzero(v).squeeze(-1)
    if v.dtype is not torch.int64:
        v = v.type(torch.int64)
    return x.index_add_(d, v, y)
