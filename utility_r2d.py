import torch
from torch.autograd import Variable
import copy
import numpy as np
import time
def normalize(x, bound, time_flag=False):
    # normalize to -1 ~ 1  (bound can be a tensor)
    #return x
    time_0 = time.time()
    bound = torch.tensor(bound)
    if len(x[0]) != len(bound):
        # then the proceding is obstacle
        # don't normalize obstacles
        # # TODO: for R2D, normalize obstacles when loading, and here we only handle input
        x[:,:-len(bound)] = x[:,:-len(bound)] / bound
        x[:,-len(bound):] = x[:,-len(bound):] / bound
    else:
        x = x / bound
    if time_flag:
        return x, time.time() - time_0
    else:
        return x
def unnormalize(x, bound, time_flag=False):
    # normalize to -1 ~ 1  (bound can be a tensor)
    # x only one dim
    #return x
    time_0 = time.time()
    bound = torch.tensor(bound)
    if len(x) != len(bound):
        # then the proceding is obstacle
        # don't normalize obstacles
        x[:,:-len(bound)] = x[:,:-len(bound)] * bound
        x[:,-len(bound):] = x[:,-len(bound):] * bound
    else:
        x = x * bound
    if time_flag:
        return x, time.time() - time_0
    else:
        return x
