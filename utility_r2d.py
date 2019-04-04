import torch
from torch.autograd import Variable
import copy
import numpy as np
def normalize(x, bound):
    # normalize to -1 ~ 1  (bound can be a tensor)
    #return x
    bound = torch.tensor(bound)
    if len(x[0]) != len(bound):
        # then the proceding is obstacle
        # don't normalize obstacles
        #x[:,:-2*len(bound)] = x[:,:-2*len(bound)] / bound[0] * 20.
        print('before normalize...')
        print(x[:,-2*len(bound):])
        x[:,-2*len(bound):-len(bound)] = x[:,-2*len(bound):-len(bound)] / bound
        x[:,-len(bound):] = x[:,-len(bound):] / bound
        print('after normalize...')
        print(x[:, -2*len(bound):])
    else:
        x = x / bound
    return x

def unnormalize(x, bound):
    # normalize to -1 ~ 1  (bound can be a tensor)
    # x only one dim
    #return x
    bound = torch.tensor(bound)
    if len(x) != len(bound):
        # then the proceding is obstacle
        # don't normalize obstacles
        #x[:-2*len(bound)] = x[:-2*len(bound)] * bound[0] / 20.
        x[-2*len(bound):-len(bound)] = x[-2*len(bound):-len(bound)] * bound
        x[-len(bound):] = x[-len(bound):] * bound
    else:
        x = x * bound
    return x
