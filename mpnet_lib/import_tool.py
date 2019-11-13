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


class fileImport():
    def __init__(self):
        self.joint_range = np.array(
            [3.4033, 3.194, 6.117, 3.6647, 6.117, 6.1083, 2.67])
        self.joint_names = ['s0', 's1', 'w0', 'w1', 'w2', 'e0', 'e1']

    def moveit_unscramble(self, paths):
        new_paths = []
        for i, path in enumerate(paths):
            new_path = np.zeros((path.shape[0], 7))
            new_path[:, 0:2] = path[:, 0:2]
            new_path[:, 2:5] = path[:, 4:]
            new_path[:, 5:] = path[:, 2:4]

            new_path = np.divide(new_path, self.joint_range)
            new_paths.append(new_path)

        return new_paths

    def paths_import_all(self, path_fname):
        with open(path_fname, "rb") as paths_f:
            paths_dict = pickle.load(paths_f)

        # env_names = paths_dict.keys() # keyed by environment name
        unscrambled_dict = {}
        for key in paths_dict.keys():
            unscrambled_dict[key] = self.moveit_unscramble(paths_dict[key])

        return unscrambled_dict

    def paths_import_single(self, path_fname, env_name, single_env=False):
        if not single_env:
            with open(path_fname, "rb") as paths_f:
                paths_dict = pickle.load(paths_f)

            env_paths = self.moveit_unscramble(paths_dict[env_name])
            return env_paths

        else:
            with open(path_fname, "rb") as paths_f:
                paths_list = pickle.load(paths_f)

            env_paths = self.moveit_unscramble(paths_list)
            return env_paths

    def pointcloud_import_array(self, pcd_fname, min_length_array):
        pc = pypcd.PointCloud.from_path(pcd_fname)

        # flatten into vector
        # obs_pc = np.zeros((3, pc.pc_data['x'].shape[0]))
        obs_pc = np.zeros((3, min_length_array))
        obs_pc[0] = pc.pc_data['x'][:min_length_array]
        obs_pc[1] = pc.pc_data['y'][:min_length_array]
        obs_pc[2] = pc.pc_data['z'][:min_length_array]

        return obs_pc

    def pointcloud_import(self, pcd_fname):
        pc = pypcd.PointCloud.from_path(pcd_fname)

        # flatten into vector
        temp = np.zeros((3, pc.pc_data['x'].shape[0]))
        temp[0] = pc.pc_data['x']
        temp[1] = pc.pc_data['y']
        temp[2] = pc.pc_data['z']

        # flattened column wise, [x0, y0, z0, x1, y1, z1, x2, y2, ...], not sure what to do about nans yet -- only fixed length if they're included
        obs_pc = temp.flatten('F')

        return obs_pc

    def pontcloud_length_check(self, pcd_fname):
        pc = self.pointcloud_import(pcd_fname)
        return pc.shape[0]

    def environments_import(self, envs_fname):
        with open(envs_fname, "rb") as env_f:
            envs = pickle.load(env_f)

        env_names = envs['poses'].keys()  # also has obstacle meta data
        return env_names

    def pointcloud_to_voxel(self, points, voxel_size=(24, 24, 24), padding_size=(32, 32, 32)):
        voxels = [self.voxelize(points[i], voxel_size, padding_size) for i in range(len(points))]
        # return size: BxV*V*V
        return np.array(voxels)

    def voxelize(self, points, voxel_size=(24, 24, 24), padding_size=(32, 32, 32), resolution=0.05):
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
