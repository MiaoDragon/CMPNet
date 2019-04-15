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
        x[:,:-2*len(bound)] = x[:,:-2*len(bound)] / bound[0]
        #print('before normalize...')
        #print(x[:,-2*len(bound):])
        x[:,-2*len(bound):-len(bound)] = x[:,-2*len(bound):-len(bound)] / bound
        x[:,-len(bound):] = x[:,-len(bound):] / bound
        #print('after normalize...')
        #print(x[:, -2*len(bound):])
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
        x[:,:-2*len(bound)] = x[:,:-2*len(bound)] * bound[0]
        x[:,-2*len(bound):-len(bound)] = x[:,-2*len(bound):-len(bound)] * bound
        x[:,-len(bound):] = x[:,-len(bound):] * bound
    else:
        x = x * bound
    if time_flag:
        return x, time.time() - time_0
    else:
        return x
