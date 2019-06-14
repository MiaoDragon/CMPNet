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
#N=number of environments; NP=Number of Paths
def load_train_dataset(N=100,NP=4000,folder='../data/simple/',s=0):
	# load data as [path]
	# for each path, it is
	# [[input],[target],[env_id]]
	obs = []
	# add start s
	for i in range(0,N):
		#load obstacle point cloud
		temp=np.fromfile(folder+'obs_cloud/obc'+str(i+s)+'.dat')
		obs.append(temp)
	obs = np.array(obs)

	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'env/'+'e'+str(i+s)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2)
				path_lengths[i][j]=len(path)
				if len(path)> max_length:
					max_length=len(path)


	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'env/'+'e'+str(i+s)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2)
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]


	dataset=[]
	targets=[]
	env_indices=[]
	for i in range(0,N):
		for j in range(0,NP):
			if path_lengths[i][j]>0:
				for m in range(0, path_lengths[i][j]-1):
					data = np.concatenate( (paths[i][j][m], paths[i][j][path_lengths[i][j]-1]) ).astype(np.float32)
					targets.append(paths[i][j][m+1])
					dataset.append(data)
					env_indices.append(i)
	# only return raw data (in order), follow below to randomly shuffle
	data=list(zip(dataset,targets,env_indices))
	random.shuffle(data)
	dataset,targets,env_indices=list(zip(*data))
	dataset = list(dataset)
	targets = list(targets)
	env_indices = list(env_indices)
	return obs, dataset, targets, env_indices
#N=number of environments; NP=Number of Paths
def load_dataset(N=100,NP=4000,folder='../data/simple/',s=0):
	# load data as [path]
	# for each path, it is
	# [[input],[target],[env_id]]
	obs = []
	# add start s
	for i in range(0,N):
		#load obstacle point cloud
		temp=np.fromfile(folder+'obs_cloud/obc'+str(i+s)+'.dat')
		obs.append(temp)
	obs = np.array(obs)

	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'env/'+'e'+str(i+s)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2)
				path_lengths[i][j]=len(path)
				if len(path)> max_length:
					max_length=len(path)


	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'env/'+'e'+str(i+s)+'/path'+str(j)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2)
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
	# only return raw data (in order), follow below to randomly shuffle
	return obs, path_data

def load_raw_dataset(N=100,NP=4000,s=0,sp=0,folder='../data/simple/'):
	obc=np.zeros((N,7,2),dtype=np.float32)
	temp=np.fromfile(folder+'obs.dat')
	obs=temp.reshape(len(temp)//2,2)

	temp=np.fromfile(folder+'obs_perm2.dat',np.int32)
	perm=temp.reshape(77520,7)

	## loading obstacles
	for i in range(0,N):
		for j in range(0,7):
			for k in range(0,2):
				obc[i][j][k]=obs[perm[i+s][j]][k]


	obs = []
	k=0
	for i in range(s,s+N):
		temp=np.fromfile(folder+'obs_cloud/obc'+str(i)+'.dat')
		obs.append(temp)
	obs = np.array(obs).astype(np.float32)
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'env/'+'e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2)
				path_lengths[i][j]=len(path)
				if len(path)> max_length:
					max_length=len(path)

	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'env/'+'e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2)
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]
	return 	obc,obs,paths,path_lengths
#N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
#Unseen_environments==> N=10, NP=2000,s=100, sp=0
#seen_environments==> N=100, NP=200,s=0, sp=4000
def load_test_dataset(N=100,NP=200, s=0,sp=4000, folder='../data/simple/'):
	obc=np.zeros((N,7,2),dtype=np.float32)
	temp=np.fromfile(folder+'obs.dat')
	obs=temp.reshape(len(temp)//2,2)

	temp=np.fromfile(folder+'obs_perm2.dat',np.int32)
	perm=temp.reshape(77520,7)

	## loading obstacles
	for i in range(0,N):
		for j in range(0,7):
			for k in range(0,2):
				obc[i][j][k]=obs[perm[i+s][j]][k]


	obs = []
	k=0
	for i in range(s,s+N):
		temp=np.fromfile(folder+'obs_cloud/obc'+str(i)+'.dat')
		obs.append(temp)
	obs = np.array(obs).astype(np.float32)
	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'env/'+'e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2)
				path_lengths[i][j]=len(path)
				if len(path)> max_length:
					max_length=len(path)


	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i in range(0,N):
		for j in range(0,NP):
			fname=folder+'env/'+'e'+str(i+s)+'/path'+str(j+sp)+'.dat'
			if os.path.isfile(fname):
				path=np.fromfile(fname)
				path=path.reshape(len(path)//2,2)
				for k in range(0,len(path)):
					paths[i][j][k]=path[k]

	return 	obc,obs,paths,path_lengths
