import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
#import nltk
#from PIL import Image
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math
import gc
from pypcd import pypcd
import numpy as np
import torch
import sys
#N=number of environments; NP=Number of Paths
def load_dataset(N=1,NP=4000,folder='../data/simple/',s=0):
    # load data as [path]
    # for each path, it is
    # [[input],[target],[env_id]]
    obs = []
    # add start s
    for i in range(0,N):
        #load obstacle point cloud
        pc = pypcd.PointCloud.from_path(folder+'home_env.pcd')
        # flatten into vector
        temp = []
        temp.append(pc.pc_data['x'][~np.isnan(pc.pc_data['x'])])
        temp.append(pc.pc_data['y'][~np.isnan(pc.pc_data['x'])])
        temp.append(pc.pc_data['z'][~np.isnan(pc.pc_data['x'])])
        temp = np.array(temp).T # N*3
        obs.append(temp)
    obs = np.array(obs)
    # add channel dim after batch dim
    obs = pointcloud_to_voxel(obs, voxel_size=[32,32,32]).reshape(-1,1,32,32,32)
    # normalize obstacle into -1~1
    #print('loading...')
    #print('original obstacle:')
    #print(obs)
    """
    lower = np.array([-383.8, -371.47, -0.2])
    higher = np.array([325, 337.89, 142.33])
    bound = (higher - lower) / 2
    obs = (obs - lower) / bound - 1.0
    print('after normalization:')
    print(obs)
    obs = obs.reshape(N,-1).astype(np.float32)
    """

    ## calculating length of the longest trajectory
    max_length=0
    path_lengths=np.zeros((N,NP),dtype=np.int8)
    for i in range(0,N):
        for j in range(0,NP):
            fname=folder+'paths/'+'path_'+str(j)+'.txt'
            if os.path.isfile(fname):
                path=np.loadtxt(fname)
                #path=path.reshape(len(path)//7,7)
                path_lengths[i][j]=len(path)
                if len(path)> max_length:
                    max_length=len(path)


    paths=np.zeros((N,NP,max_length,7), dtype=np.float32)   ## padded paths

    for i in range(0,N):
        for j in range(0,NP):
            fname=folder+'paths/'+'path_'+str(j)+'.txt'
            if os.path.isfile(fname):
                path=np.loadtxt(fname)
                print('loaded path')
                print(path.shape)
                #path=path.reshape(len(path)//7,7)
                for k in range(0,len(path)):
                    paths[i][j][k]=path[k]



    path_data = []
    for i in range(0,N):
        for j in range(0,NP):
            dataset=[]
            targets=[]
            env_indices=[]
            if path_lengths[i][j]>0:
                for m in range(0, path_lengths[i][j]-1):
                    for n in range(m+1, path_lengths[i][j]):
                        #data = np.concatenate( (paths[i][j][m], paths[i][j][path_lengths[i][j]-1]) ).astype(np.float32)
                        #targets.append(paths[i][j][m+1])
                        #dataset.append(data)
                        #env_indices.append(i)
                        # forward
                        data = np.concatenate( (paths[i][j][m], paths[i][j][n]) ).astype(np.float32)
                        targets.append(paths[i][j][m+1])

                        dataset.append(data)
                        env_indices.append(i)
                        # backward
                        data = np.concatenate( (paths[i][j][n], paths[i][j][m]) ).astype(np.float32)
                        targets.append(paths[i][j][n-1])

                        dataset.append(data)
                        env_indices.append(i)
            path_data.append([dataset, targets, env_indices])
    print('after loading.')
    print('shape:')
    print('obstacle')
    print(obs.shape)
    print('path')
    print(len(path_data))
    # only return raw data (in order), follow below to randomly shuffle
    return obs, path_data
    # data=list(zip(dataset,targets))
    # random.shuffle(data)
    # dataset,targets=list(zip(*data))
    # dataset and targets are both list
    # here the first item of data is index in obs
    # return obs, list(zip(*data))
#N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
#Unseen_environments==> N=10, NP=2000,s=100, sp=0
#seen_environments==> N=100, NP=200,s=0, sp=4000
def load_test_dataset(N=100,NP=200, s=0,sp=4000, folder='../data/simple/'):
    obc = [None for i in range(N)]
    obs = []
    # add start s
    for i in range(0,N):
        #load obstacle point cloud
        pc = pypcd.PointCloud.from_path(folder+'home_env.pcd')
        # flatten into vector
        temp = []
        temp.append(pc.pc_data['x'][~np.isnan(pc.pc_data['x'])])
        temp.append(pc.pc_data['y'][~np.isnan(pc.pc_data['x'])])
        temp.append(pc.pc_data['z'][~np.isnan(pc.pc_data['x'])])
        temp = np.array(temp).T # N*3
        obs.append(temp)
    obs = np.array(obs)
    obs = pointcloud_to_voxel(obs, voxel_size=[32,32,32]).reshape(-1,1,32,32,32)

    #print('loading...')
    #print('original obstacle:')
    #print(obs)
    """
    if N > 0:
        lower = np.array([-383.8, -371.47, -0.2])
        higher = np.array([325, 337.89, 142.33])
        bound = (higher - lower) / 2
        obs = (obs - lower) / bound - 1.0
        print('after normalization:')
        print(obs)
        obs = obs.reshape(N, -1).astype(np.float32)
    """

    ## calculating length of the longest trajectory
    max_length=0
    path_lengths=np.zeros((N,NP),dtype=np.int8)
    for i in range(0,N):
        for j in range(0,NP):
            fname=folder+'paths/'+'path_'+str(j+sp)+'.txt'
            if os.path.isfile(fname):
                path=np.loadtxt(fname)
                #path=path.reshape(len(path)//7,7)
                path_lengths[i][j]=len(path)
                if len(path)> max_length:
                    max_length=len(path)

    paths=np.zeros((N,NP,max_length,7), dtype=np.float32)   ## padded paths

    for i in range(0,N):
        for j in range(0,NP):
            fname=folder+'paths/'+'path_'+str(j+sp)+'.txt'
            if os.path.isfile(fname):
                path=np.loadtxt(fname)
                print('loading path...')
                print(path.shape)
                #path=path.reshape(len(path)//7,7)
                for k in range(0,len(path)):
                    paths[i][j][k]=path[k]

    print("after loading...")
    print("obstacle")
    print(obs.shape)
    print('paths:')
    print(paths.shape)
    return obc,obs,paths,path_lengths


def pointcloud_to_voxel(points, voxel_size=(24, 24, 24), padding_size=(32, 32, 32)):
    voxels = [voxelize(points[i], voxel_size, padding_size) for i in range(len(points))]
    # return size: BxV*V*V
    return np.array(voxels)

def voxelize(points, voxel_size=(24, 24, 24), padding_size=(32, 32, 32), resolution=0.05):
    """
    Convert `points` to centerlized voxel with size `voxel_size` and `resolution`, then padding zero to
    `padding_to_size`. The outside part is cut, rather than scaling the points.

    Args:
    `points`: pointcloud in 3D numpy.ndarray (shape: N * 3)
    `voxel_size`: the centerlized voxel size, default (24,24,24)
    `padding_to_size`: the size after zero-padding, default (32,32,32)
    `resolution`: the resolution of voxel, in meters

    Ret:
    `voxel`:32*32*32 voxel occupany grid
    `inside_box_points`:pointcloud inside voxel grid
    """
    # calculate resolution based on boundary
    if abs(resolution) < sys.float_info.epsilon:
        print('error input, resolution should not be zero')
        return None, None

    """
    here the point cloud is centerized, and each dimension uses a different resolution
    """
    OCCUPIED = 1
    FREE = 0
    resolution = [(points[:,i].max() - points[:,i].min()) / voxel_size[i] for i in range(3)]
    resolution = np.array(resolution)
    #resolution = np.max(res)
    # remove all non-numeric elements of the said array
    points = points[np.logical_not(np.isnan(points).any(axis=1))]

    # filter outside voxel_box by using passthrough filter
    # TODO Origin, better use centroid?
    origin = (np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2]))
    # set the nearest point as (0,0,0)
    points[:, 0] -= origin[0]
    points[:, 1] -= origin[1]
    points[:, 2] -= origin[2]
    # logical condition index
    x_logical = np.logical_and((points[:, 0] < voxel_size[0] * resolution[0]), (points[:, 0] >= 0))
    y_logical = np.logical_and((points[:, 1] < voxel_size[1] * resolution[1]), (points[:, 1] >= 0))
    z_logical = np.logical_and((points[:, 2] < voxel_size[2] * resolution[2]), (points[:, 2] >= 0))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    inside_box_points = points[xyz_logical]

    # init voxel grid with zero padding_to_size=(32*32*32) and set the occupany grid
    voxels = np.zeros(padding_size)
    # centerlize to padding box
    center_points = inside_box_points + (padding_size[0] - voxel_size[0]) * resolution / 2
    # TODO currently just use the binary hit grid
    x_idx = (center_points[:, 0] / resolution[0]).astype(int)
    y_idx = (center_points[:, 1] / resolution[1]).astype(int)
    z_idx = (center_points[:, 2] / resolution[2]).astype(int)
    voxels[x_idx, y_idx, z_idx] = OCCUPIED
    return voxels
    #return voxels, inside_box_points
