import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
# import nltk
from PIL import Image
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math
import pypcd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

joint_range = np.array([3.4033, 3.194, 6.117, 3.6647, 6.117, 6.1083, 2.67])
joint_names = ['s0', 's1', 'w0', 'w1', 'w2', 'e0', 'e1']

def scale_path(path):
    path = np.multiply(path, joint_range)
    return path

def pc2np(pc):
    np_pc = np.zeros((3, pc.pc_data['x'].shape[0]))

    np_pc[0] = pc.pc_data['x']
    np_pc[1] = pc.pc_data['y']
    np_pc[2] = pc.pc_data['z']

    return np_pc

def torch2np(pcVariable):
    obs = pcVariable.data.cpu()
    obs = obs.numpy()
    return obs

def pointcloud_reconstruct(pc_flat):
    length = pc_flat.shape[0]
    pc_recon = pc_flat.reshape((3, length/3), order='F')

    return pc_recon

def decoder_reconstruct(decoder_output):
    output = torch2np(decoder_output)
    pc_recon = pointcloud_reconstruct(output)

    return pc_recon

def plot_pointcloud(pc_array, figsize=(7,5)):
    fig = plt.figure(figsize=figsize, )
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_array[0, :], pc_array[1, :], pc_array[2, :], color='b')
    ax.view_init(30, 30)
    return fig, ax
