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
from Model.AE.data_loader_baxter import load_normalized_dataset

#N=number of environments; NP=Number of Paths


def load_dataset(env_names, data_path, pcd_path, importer, NP=940, min_length=5351*3):
	N = len(env_names)
	obs = load_normalized_dataset(env_names, pcd_path, importer)

	### obtain path length data ###
	# paths_file = 'trainPathsLarge.pkl'
	paths_file = 'trainPathsLarge_RRTSTAR_Fix.pkl'

	# calculating length of the longest trajectory
	max_length = 0
	path_lengths = np.zeros((N, NP), dtype=np.int64)
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(
			path_fname=data_path+paths_file, env_name=env, single_env=False)
		for j in range(0, NP):  # for j in num_paths:
			path_lengths[i][j] = len(env_paths[j])
			if len(env_paths[j]) > max_length:
				max_length = len(env_paths[j])

	# padded paths #7D from 2D originally
	paths = np.zeros((N, NP, max_length, 7), dtype=np.float32)

	for i in range(0, N):
		env_paths = importer.paths_import_single(
			path_fname=data_path+paths_file, env_name=env, single_env=False)
		for j in range(0, NP):
			paths[i][j][:len(env_paths[j])] = env_paths[j]

	# clean up paths
	paths = paths[:, :, 1:, :]
	path_lengths = path_lengths - 1
	print("Imported path data")
	print(paths.shape)

	path_data = []
	for i in range(0, N):
		for j in range(0, NP):
			dataset = []
			targets = []
			env_indices = []
			if path_lengths[i][j] > 0:
				for m in range(0, path_lengths[i][j]-1):
					data = np.concatenate(
						(paths[i][j][m], paths[i][j][path_lengths[i][j]-1])).astype(np.float32)
					targets.append(paths[i][j][m+1])
					dataset.append(data)
					env_indices.append(i)
			path_data.append([dataset, targets, env_indices])

	return obs, path_data


def load_raw_dataset(env_names, data_path, pcd_path, importer, NP=940, min_length=5351*3):
	N = len(env_names)
	obs = load_normalized_dataset(env_names, pcd_path, importer)

	### obtain path length data ###
	# paths_file = 'trainPathsLarge.pkl'
	# paths_file = 'trainPathsLarge_GoalsCorrect_RRTSTAR.pkl'
	paths_file = 'trainPathsLarge_GoalsCorrect_RRTSTAR_trainEnv_4.pkl'

	# calculating length of the longest trajectory
	print("getting path lengths...")
	max_length = 0
	path_lengths = np.zeros((N, NP), dtype=np.int64)
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(
			path_fname=data_path+paths_file, env_name=env, single_env=True)
		for j in range(0, NP):  # for j in num_paths:
			path_lengths[i][j] = len(env_paths[j])
			if len(env_paths[j]) > max_length:
				max_length = len(env_paths[j])

	# padded paths #7D from 2D originally
	paths = np.zeros((N, NP, max_length, 7), dtype=np.float32)

	print("loading paths...")
	for i in range(0, N):
		env_paths = importer.paths_import_single(
			path_fname=data_path+paths_file, env_name=env, single_env=True)
		for j in range(0, NP):
			paths[i][j][:len(env_paths[j])] = env_paths[j]

	# clean up paths
	paths = paths[:, :, 1:, :]
	path_lengths = path_lengths - 1
	return obs, paths, path_lengths
	
#N=number of environments; NP=Number of Paths; s=starting environment no.; sp=starting_path_no
#Unseen_environments==> N=10, NP=2000,s=100, sp=0
#seen_environments==> N=100, NP=200,s=0, sp=4000


def load_test_dataset(env_names, data_path, pcd_path, importer, NP=100, min_length=5351*3):
	N = len(env_names)
	obstacles = load_normalized_dataset(env_names, pcd_path, importer)

	### obtain path length data ###
	# paths_file = 'trainEnvironments_testPaths_GoalsCorrect_RRTSTAR_trainEnv_4.pkl'
	paths_file = 'trainEnvironments_testPaths_GoalsCorrect_RRTSTAR.pkl'
	print("LOADING FROM: ")
	print(paths_file)
	# calculating length of the longest trajectory
	max_length = 0
	path_lengths = np.zeros((N, NP), dtype=np.int64)
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(
			path_fname=data_path+paths_file, env_name=env, single_env=False)
		print("env len: " + str(len(env_paths)))
		print("i: " + str(i))
		print("env name: " + env)
		for j in range(0, NP):  # for j in num_paths:
			path_lengths[i][j] = len(env_paths[j])
			if len(env_paths[j]) > max_length:
				max_length = len(env_paths[j])

	# padded paths #7D from 2D originally
	paths = np.zeros((N, NP, max_length, 7), dtype=np.float32)
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(
			path_fname=data_path+paths_file, env_name=env, single_env=False)
		for j in range(0, NP):
			paths[i][j][:len(env_paths[j])] = env_paths[j]

	# clean up paths
	paths_new = paths[:, :, 1:, :]
	path_lengths_new = path_lengths - 1

	return obstacles, paths_new, path_lengths_new
