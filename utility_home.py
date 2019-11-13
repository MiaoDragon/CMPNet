import torch
from torch.autograd import Variable
import copy
import numpy as np
import time
def normalize(x, bound, time_flag=False):
    # normalize to -1 ~ 1  (bound can be a tensor)
    #return x
    time_0 = time.time()
    lower = np.array([-383.8, -371.47, -0.2, -1, -1, -1, -1])
    higher = np.array([325, 337.89, 142.33, 1, 1, 1, 1])
    bound = (higher - lower) / 2
    bound = torch.from_numpy(bound).type(torch.FloatTensor)
    lower = torch.from_numpy(lower).type(torch.FloatTensor)
    higher = torch.from_numpy(higher).type(torch.FloatTensor)
    #print('before normalizing...')
    #print(x)
    if len(x.size()) != 1:
        # then the proceding is obstacle
        # don't normalize obstacles
        # we assume the obstacle pcd has been normalized
        x[:,:-len(bound)] = (x[:,:-len(bound)]-lower) / bound - 1.0
        x[:,-len(bound):] = (x[:,-len(bound):]-lower) / bound - 1.0
        #print('after normalizing...')
        #print(x[:,-2*len(bound):])
    else:
        #print('before normalizing...')
        #print(x)
        if len(x) == len(bound):
            x = (x - lower) / bound - 1.0
        else:
            x[:-len(bound)] = (x[:-len(bound)]-lower) / bound - 1.0
            x[-len(bound):] = (x[-len(bound):]-lower) / bound - 1.0
        #print('after normalizing...')
        #print(x)
    if time_flag:
        return x, time.time() - time_0
    else:
        return x
def unnormalize(x, bound, time_flag=False):
    # from -1~1 to the actual bound
    # x only one dim
    #return x
    time_0 = time.time()
    lower = np.array([-383.8, -371.47, -0.2, -1, -1, -1, -1])
    higher = np.array([325, 337.89, 142.33, 1, 1, 1, 1])
    bound = (higher - lower) / 2
    bound = torch.from_numpy(bound).type(torch.FloatTensor)
    lower = torch.from_numpy(lower).type(torch.FloatTensor)
    higher = torch.from_numpy(higher).type(torch.FloatTensor)

    if len(x.size()) != 1:
        # then the proceding is obstacle
        # don't normalize obstacles
        #x[:,:-2*len(bound)] = x[:,:-2*len(bound)] * bound[0]
        #print('before unnormalizing...')
        #print(x[:, -2*len(bound):])
        x[:,:-len(bound)] = (x[:,:-len(bound)] + 1.0) * bound + lower
        x[:,-len(bound):] = (x[:,-len(bound):] + 1.0) * bound + lower
        #print('after unnormalizing...')
        #print(x[:, -2*len(bound):])
    else:
        #print('before unnormalizing...')
        #print(x)
        if len(x) == len(bound):
            x = (x + 1.0) * bound + lower
        else:
            x[:-len(bound)] = (x[:-len(bound)] + 1.0) * bound + lower
            x[-len(bound):] = (x[-len(bound):] + 1.0) * bound + lower
        #print('after unnormalizing...')
        #print(x)
    if time_flag:
        return x, time.time() - time_0
    else:
        return x
