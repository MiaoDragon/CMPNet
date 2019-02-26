import torch
from torch.autograd import Variable
import copy
def normalize(x, bound):
    # normalize to -1 ~ 1  (bound can be a tensor)
    return x / np.array(bound)
def unnormalize(x, bound):
    return x * np.array(bound)
