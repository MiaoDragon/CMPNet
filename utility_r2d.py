import torch
from torch.autograd import Variable
import copy
import numpy as np
def normalize(x, bound):
    # normalize to -1 ~ 1  (bound can be a tensor)
    return x / torch.tensor(bound)
def unnormalize(x, bound):
    return x * torch.tensor(bound)
