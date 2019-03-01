import torch
from torch.autograd import Variable
import copy

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def save_state(net, torch_seed, np_seed, py_seed, fname):
    # save both model state and optimizer state
    states = {
        'state_dict': net.state_dict(),
        'optimizer': net.opt.state_dict(),
        'torch_seed': torch_seed,
        'np_seed': np_seed,
        'py_seed': py_seed
    }
    torch.save(states, fname)

def load_net_state(net, fname):
    checkpoint = torch.load(fname, map_location=torch.cuda.current_device())
    net.load_state_dict(checkpoint['state_dict'])

def load_opt_state(net, fname):
    checkpoint = torch.load(fname, map_location=torch.cuda.current_device())
    net.opt.load_state_dict(checkpoint['optimizer'])

def load_seed(fname):
    # load both torch random seed, and numpy random seed
    checkpoint = torch.load(fname, map_location=torch.cuda.current_device())
    return checkpoint['torch_seed'], checkpoint['np_seed'], checkpoint['py_seed']
