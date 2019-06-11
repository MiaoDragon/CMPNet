import torch
from torch.autograd import Variable
import copy
import numpy as np

import sys
sys.path.append('/media/arclabdl1/HD1/Anthony/repos/baxter_mpnet/scripts/')
# sys.path.append('/baxter_mpnet_docker/scripts')
from motion_planning_dataset import *
from moveit_functions import *
from get_state_validity import *


def BaxterIsInCollision(x, obc):
    global filler_robot_state
    global rs_man
    global sv

    joint_ranges = np.array(
        [3.4033, 3.194, 6.117, 3.6647, 6.117, 6.1083, 2.67])

    filler_robot_state[10:17] = moveit_scrambler(np.multiply(x, joint_ranges))
    rs_man.joint_state.position = tuple(filler_robot_state)

    collision_free = sv.getStateValidity(rs_man, group_name="right_arm")
    return not collision_free

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
    checkpoint = torch.load(fname, map_location='cpu')
    # checkpoint = torch.load(fname)
    net.load_state_dict(checkpoint['state_dict'])


def load_opt_state(net, fname):
    checkpoint = torch.load(fname, map_location='cpu')
    # checkpoint = torch.load(fname)
    net.opt.load_state_dict(checkpoint['optimizer'])


def load_seed(fname):
    # load both torch random seed, and numpy random seed
    checkpoint = torch.load(fname, map_location='cpu')
    # checkpoint = torch.load(fname)
    return checkpoint['torch_seed'], checkpoint['np_seed'], checkpoint['py_seed']


# def IsInCollision(x, obc, filler_robot_state, sv, rs_man):
#     joint_ranges = np.array([3.4033, 3.194, 6.117, 3.6647, 6.117, 6.1083, 2.67])

#     filler_robot_state[10:17] = moveit_scrambler(np.multiply(x, joint_ranges))
#     rs_man.joint_state.position = tuple(filler_robot_state)

#     collision_free = sv.getStateValidity(rs_man, group_name="right_arm")
#     return collision_free
