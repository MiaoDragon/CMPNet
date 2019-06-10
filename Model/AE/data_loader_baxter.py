import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import os.path
import random
import fnmatch
import pickle


def load_dataset(task, N=30000,NP=1800):

	obstacles=np.zeros((N,2800),dtype=np.float32)
	for i in range(0,N):
		if task == 0:
			name = 'simple'
		else:
			name = 'complex'
		temp=np.fromfile(('../data/%s/obs_cloud/obc' % (name))+str(i)+'.dat')
		temp=temp.reshape(len(temp)/2,2)
		obstacles[i]=temp.flatten()


	return 	obstacles


# ,N=30000,NP=1800):
def load_normalized_dataset(env_names, data_path, importer, min_length=(5351*3)):
	# get file names, just grabbing first one available (sometimes there's multiple)
	fnames = []

	print("Loading point cloud data for envs: ")
	print(env_names)

	print("Searing for file names...")
	for i, env in enumerate(env_names):
		# hacky reordering so that we don't load the last .pcd file which is always corrupt
		# sort by the time step on the back, hopefully that helps it obtain the earliest possible
		for file in sorted(os.listdir(data_path), key=lambda x: int(x.split('Env_')[1].split('_')[1][:-4])):
			if (fnmatch.fnmatch(file, env+"*")):
				fnames.append(file)
				break

	if min_length is None:  # compute minimum length for dataset will take twice as long if necessary
		min_length = 1e6  # large to start
		for i, fname in enumerate(fnames):
			length = importer.pointcloud_length_check(data_path + fname)
			if (length < min_length):
				min_length = length

	N = len(fnames)

	min_length_array = min_length//3
	obstacles_array = np.zeros((3, min_length_array, N), dtype=np.float32)
	for i, fname in enumerate(fnames):
		data = importer.pointcloud_import_array(data_path + fname, min_length_array)
		obstacles_array[:, :, i] = data

	# compute mean and std of each environment
	means = np.mean(obstacles_array, axis=1)
	stds = np.std(obstacles_array, axis=1)
	norms = np.linalg.norm(obstacles_array, axis=1)

	# compute mean and std of means and stds
	mean_overall = np.expand_dims(np.mean(means, axis=1), axis=1)
	std_overall = np.expand_dims(np.std(stds, axis=1), axis=1)
	norm_overall = np.expand_dims(np.mean(norms, axis=1), axis=1)

	obstacles = np.zeros((N, min_length), dtype=np.float32)
	for i in range(obstacles_array.shape[2]):
		temp_arr = (obstacles_array[:, :, i] - mean_overall)
		temp_arr = np.divide(temp_arr, norm_overall)
		obstacles[i] = temp_arr.flatten('F')

	return obstacles
