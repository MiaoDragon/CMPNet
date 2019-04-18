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
from mpnet_lib.import_tool import fileImport
import fnmatch
from data_loader_baxter import load_normalized_dataset

# Environment Encoder
class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(16053, 786), nn.PReLU(),
									 nn.Linear(786, 512), nn.PReLU(),
									 nn.Linear(512, 256),nn.PReLU(),
									 nn.Linear(256, 60))

	def forward(self, x):
		x = self.encoder(x)
		return x

#N=number of environments; NP=Number of Paths, environments = list of environment names
def load_dataset(env_names,data_path,pcd_path,importer,cae_path,NP=940,min_length=5351*3): #N=100,NP=4000, environments):

	Q = Encoder()
	Q.load_state_dict(torch.load(cae_path)) #we don't have this yet
	if torch.cuda.is_available():
		Q.cuda()

	print("Encoder network: ")
	print(Q)

	env_fnames = []
	paths_file = 'trainPathsLarge.pkl'

	# for i, env in enumerate(env_names):
	# 	for file in sorted(os.listdir(pcd_path), key=lambda x: int(x.split('Env_')[1].split('_')[1][:-4])):
	# 	# for file in os.listdir(data_path):
	# 	# 	f_string = str(file)
	# 	# 	print(f_string.split('trainEnv_'))
	# 		if (fnmatch.fnmatch(file, env+"*")):
	# 			env_fnames.append(file)
	# 			break

	N = len(env_names)
	print(N)

	print("importing normalized point cloud dataset\n")
	obs_rep=np.zeros((N,60),dtype=np.float32)
	obstacles = load_normalized_dataset(env_names, pcd_path, importer)
	inp=torch.from_numpy(obstacles)
	inp=Variable(inp.float()).cuda()
	output=Q(inp)
	output=output.data.cpu()
	obs_rep=output.numpy()
	print("obstacle representation array: ")
	print(obs_rep.shape)

	# for i, fname in enumerate(env_fnames): #for i, env in enumerate(environments)
	# 	#load obstacle point cloud
	# 	obstacles=np.zeros((1,min_length),dtype=np.float32)
	# 	data = importer.pointcloud_import(pcd_path + fname)
	# 	obstacles[0] = data[:min_length]
	# 	inp=torch.from_numpy(obstacles)
	# 	inp=Variable(inp.float()).cuda()
	# 	output=Q(inp)
	# 	output=output.data.cpu()
	# 	obs_rep[i]=output.numpy()

	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int64)
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(path_fname=data_path+paths_file, env_name=env)
		for j in range(0,NP): #for j in num_paths:
			path_lengths[i][j] = len(env_paths[j])
			if len(env_paths[j])> max_length:
				max_length=len(env_paths[j])

	print("Obtained max path length: \n")
	print(max_length)
	paths=np.zeros((N,NP,max_length,7), dtype=np.float32)   ## padded paths #7D from 2D originally

	for i in range(0,N):
		env_paths = importer.paths_import_single(path_fname=data_path+paths_file, env_name=env)
		for j in range(0,NP):
			paths[i][j][:len(env_paths[j])] = env_paths[j]

	print("Obtained paths,for envs: ")
	print(len(paths))
	print("Path matrix shape: ")
	print(paths.shape)
	print("\n")

	dataset=[]
	targets=[]
	# obs_rep_sz = obs_rep[0].shape[0] ?
	for i, env in enumerate(env_names):
		for j in range(0,NP):
			if path_lengths[i][j]>0:
				for m in range(0, path_lengths[i][j]-1):
					data=np.zeros(60+14,dtype=np.float32)
					for k in range(0,60):
						data[k]=obs_rep[i][k]

					for joint in range(7):
						data[60 + joint] = paths[i][j][m][joint]
						data[60 + joint + 7] = paths[i][j][path_lengths[i][j] - 1][joint]

					targets.append(paths[i][j][m+1])
					dataset.append(data)

	# clean up paths
	paths_new = paths[:, :, 1:, :]
	targets_new = targets[1:]
	dataset_new = dataset[1:]
	path_lengths_new = path_lengths - 1

	print("Length of dataset and targets: ")
	print(str(len(dataset)) + " " + str(len(targets)))
	print(str(len(dataset_new)) + " " + str(len(targets_new)))

	data=zip(dataset_new,targets_new)
	random.shuffle(data)
	dataset,targets=zip(*data)
	return 	np.asarray(dataset),np.asarray(targets)

def load_dataset_single_paths_only(env_name,data_path,importer,NP=940): #N=100,NP=4000, environments):

	env_fnames = []
	paths_file = 'trainPathsLarge.pkl'

	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((1,NP),dtype=np.int64)
	env_paths = importer.paths_import_single(path_fname=data_path+paths_file, env_name=env_name)
	for j in range(0,NP): #for j in num_paths:
		path_lengths[0][j] = len(env_paths[j])
		if len(env_paths[j])> max_length:
			max_length=len(env_paths[j])

	print("Obtained max path length: \n")
	print(max_length)
	paths=np.zeros((1,NP,max_length,7), dtype=np.float32)   ## padded paths #7D from 2D originally

	env_paths = importer.paths_import_single(path_fname=data_path+paths_file, env_name=env_name)
	for j in range(0,NP):
		paths[0][j][:len(env_paths[j])] = env_paths[j]

	print("Obtained paths,for envs: ")
	print(len(paths))
	print("Path matrix shape: ")
	print(paths.shape)
	print("\n")

	dataset=[]
	targets=[]
	# obs_rep_sz = obs_rep[0].shape[0] ?
	for j in range(0,NP):
		if path_lengths[0][j]>0:
			for m in range(0, path_lengths[0][j]-1):
				data=np.zeros(14,dtype=np.float32)

				for joint in range(7):
					data[joint] = paths[0][j][m][joint]
					data[joint + 7] = paths[0][j][path_lengths[0][j] - 1][joint]

				targets.append(paths[0][j][m+1])
				dataset.append(data)

	# clean up paths
	paths_new = paths[:, :, 1:, :]
	targets_new = targets[1:]
	dataset_new = dataset[1:]
	path_lengths_new = path_lengths - 1

	print("Length of dataset and targets: ")
	print(str(len(dataset)) + " " + str(len(targets)))
	print(str(len(dataset_new)) + " " + str(len(targets_new)))

	dataset_clean = dataset_new
	targets_clean = targets_new
	data=zip(dataset_new,targets_new)
	random.shuffle(data)
	dataset_shuffle,targets_shuffle=zip(*data)
	return 	np.asarray(dataset_shuffle), np.asarray(targets_shuffle), dataset_clean, targets_clean, paths_new, path_lengths_new

def load_dataset_end2end(env_names,data_path,pcd_path,importer,NP=940,min_length=5351*3):
	N = len(env_names)
	obstacles = load_normalized_dataset(env_names,pcd_path,importer)

	### obtain path length data ###
	paths_file = 'trainPathsLarge.pkl'
	# calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int64)
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(path_fname=data_path+paths_file, env_name=env, single_env=False)
		print("env len: " + str(len(env_paths)))
		print("i: " + str(i))
		for j in range(0,NP): #for j in num_paths:
			path_lengths[i][j] = len(env_paths[j])
			if len(env_paths[j])> max_length:
				max_length=len(env_paths[j])

	print("Obtained max path length: \n")
	print(max_length)

	### obtain path data ###

	paths=np.zeros((N,NP,max_length,7), dtype=np.float32)   ## padded paths #7D from 2D originally
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(path_fname=data_path+paths_file, env_name=env, single_env=False)
		for j in range(0,NP):
			paths[i][j][:len(env_paths[j])] = env_paths[j]

	print("Obtained paths,for envs: ")
	print(len(paths))
	print("Path matrix shape: ")
	print(paths.shape)
	print("\n")

	### create dataset and targets ###

	dataset=[]
	targets=[]
	pointcloud_inds=[]
	# obs_rep_sz = obs_rep[0].shape[0] ?
	for i, env in enumerate(env_names):
		for j in range(0,NP):
			if path_lengths[i][j]>0:
				for m in range(0, path_lengths[0][j]-1):
					data=np.zeros(14,dtype=np.float32)

					for joint in range(7):
						data[joint] = paths[i][j][m][joint]
						data[joint + 7] = paths[i][j][path_lengths[i][j] - 1][joint]

					pointcloud_inds.append(i)
					targets.append(paths[i][j][m+1])
					dataset.append(data)

	# clean up paths
	paths_new = paths[:, :, 1:, :]
	targets_new = targets[1:]
	dataset_new = dataset[1:]
	pointcloud_inds_new = pointcloud_inds[1:]
	path_lengths_new = path_lengths - 1

	# print("Length of dataset, targets, and pointclouds: ")
	# print(str(len(dataset)) + " " + str(len(targets)) + " " + str(len(pointclouds))))
	# print(str(len(dataset_new)) + " " + str(len(targets_new)) + " " + str(len(pointclouds_new))))

	# dataset_clean = dataset_new
	# targets_clean = targets_new
	# pointclouds_clean = pointclouds_new

	data=zip(dataset_new, targets_new, pointcloud_inds_new)
	random.shuffle(data)
	dataset_shuffle, targets_shuffle, pointclouds_inds_shuffle=zip(*data)

	return 	np.asarray(dataset_shuffle), np.asarray(targets_shuffle), np.asarray(pointclouds_inds_shuffle), obstacles # , dataset_clean, targets_clean, paths_new, path_lengths_new



def load_test_dataset_end2end(env_names,data_path,pcd_path,importer,NP=80,min_length=5351*3):
	N = len(env_names)
	obstacles = load_normalized_dataset(env_names,pcd_path,importer)

	### obtain path length data ###
	# paths_file = 'trainEnvironments_testPaths_GoalsCorrect_RRTSTAR_trainEnv_4.pkl'
	# paths_file = 'trainEnvironments_testPaths_GoalsCorrect_RRTSTAR.pkl'
	
	paths_file = 'trainPathsLarge_GoalsCorrect_RRTSTAR_trainEnv_4.pkl' #TRAINING DATA SANITY CHECK
	# paths_file = 'trainPathsLarge_RRTSTAR_Fix.pkl' #TRAINING DATA SANITY CHECK
	print("LOADING FROM: ")
	print(paths_file)
	# calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int64)
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(path_fname=data_path+paths_file, env_name=env, single_env=True)
		print("env len: " + str(len(env_paths)))
		print("i: " + str(i))
		print("env name: " + env)
		for j in range(0,NP): #for j in num_paths:
			path_lengths[i][j] = len(env_paths[j])
			if len(env_paths[j])> max_length:
				max_length=len(env_paths[j])

	print("Obtained max path length: \n")
	print(max_length)

	### obtain path data ###

	paths=np.zeros((N,NP,max_length,7), dtype=np.float32)   ## padded paths #7D from 2D originally
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(path_fname=data_path+paths_file, env_name=env, single_env=True)
		for j in range(0,NP):
			paths[i][j][:len(env_paths[j])] = env_paths[j]

	print("Obtained paths,for envs: ")
	print(len(paths))
	print("Path matrix shape: ")
	print(paths.shape)
	print("\n")

	### create dataset and targets ###

	# clean up paths
	paths = paths[:, :, 1:, :]
	path_lengths = path_lengths - 1

	return 	obstacles, paths, path_lengths


#N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
#Unseen_environments==> N=10, NP=2000,s=100, sp=0
#seen_environments==> N=100, NP=200,s=0, sp=4000
def load_test_dataset(env_names,data_path,importer,NP=10): #N=100,NP=200, s=0,sp=4000):

	N = len(env_names)

	Q = Encoder()
	Q.load_state_dict(torch.load('../models/cae_encoder.pkl'))
	if torch.cuda.is_available():
		Q.cuda()


	env_fnames = []
	paths_file = 'trainPaths.pkl'

	for i, env in enumerate(env_names):
		for file in os.listdir(data_path):
			if (fnmatch.fnmatch(file, env+"*")):
				env_fnames.append(file)
				break

	obs_rep=np.zeros((N,28),dtype=np.float32)
	k=0
	for i in range(s,s+N):
		#load obstacle point cloud
		obstacles=np.zeros((1,921600),dtype=np.float32)
		obstacles = importer.pointcloud_import(data_path + fname)
		inp=torch.from_numpy(obstacles)
		# inp=Variable(inp).cuda()
		inp=Variable(inp.float())
		output=Q(inp)
		output=output.data.cpu()
		obs_rep[k]=output.numpy()
		k=k+1

	## calculating length of the longest trajectory
	max_length=0
	path_lengths=np.zeros((N,NP),dtype=np.int8)
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(path_fname=data_path+paths_file, env_name=env)
		for j in range(0,NP): #for j in num_paths:
			path_lengths[i][j] = len(env_paths[j])
			if len(env_paths[j])> max_length:
				max_length=len(env_paths[j])


	paths=np.zeros((N,NP,max_length,2), dtype=np.float32)   ## padded paths

	for i, env in enumerate(env_names):
		for j in range(0,NP):
			if path_lengths[i][j]>0:
				for m in range(0, path_lengths[i][j]-1):
					data=np.zeros(28+14,dtype=np.float32)
					for k in range(0,28):
						data[k]=obs_rep[i][k]

					for joint in range(7):
						data[28 + joint] = paths[i][j][m][joint]
						data[28 + joint + 7] = paths[i][j][path_lengths[i][j] - 1][joint]

					targets.append(paths[i][j][m+1])
					dataset.append(data)





	return 	obc,obs_rep,paths,path_lengths

def main():
	importer = fileImport()
	data_path = '/home/arclab-testchamber/ros_ws/src/baxter_mpnet/data/full_dataset_sample/'

	envs_file = 'trainEnvironments.pkl'

	envs = importer.environments_import(data_path + envs_file)
	dataset,targets= load_dataset(envs,data_path,importer)

	print("Obtained dataset, length: ")
	print(len(dataset))
	print("Obtained targets, length: ")
	print(len(targets))


if __name__ == '__main__':
	main()