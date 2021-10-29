# Extensions for complex operations

import torch
from ..build.matcharrays import *

def bmul(x, y):
    x, y = matchviews(x, y)
    return torch.mul(x, y)
