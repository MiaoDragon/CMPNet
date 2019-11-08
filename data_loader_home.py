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
    # normalize obstacle into -1~1
    print('loading...')
    print('original obstacle:')
    print(obs)
    lower = np.array([-383.8, -371.47, -0.2])
    higher = np.array([325, 337.89, 142.33])
    bound = (higher - lower) / 2
    obs = (obs - lower) / bound - 1.0
    print('after normalization:')
    print(obs)
    obs = obs.reshape(N,-1)


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
                    data = np.concatenate( (paths[i][j][m], paths[i][j][path_lengths[i][j]-1]) ).astype(np.float32)
                    targets.append(paths[i][j][m+1])
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
    obc = None
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
    print('loading...')
    print('original obstacle:')
    print(obs)
    lower = np.array([-383.8, -371.47, -0.2])
    higher = np.array([325, 337.89, 142.33])
    bound = (higher - lower) / 2
    obs = (obs - lower) / bound - 1.0
    print('after normalization:')
    print(obs)
    obs = obs.reshape(N, -1)
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
