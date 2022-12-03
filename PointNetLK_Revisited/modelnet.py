
import os
import glob
import numpy as np
import torch
import torch.utils.data
import six
import copy
import csv
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

import utils
from data_utils import add_noise, find_voxel_overlaps, points_to_voxel_second


# by Rajat Talak
class PointRegistrationModelNet(torch.utils.data.Dataset):
    def __init__(self, args, dataset, sigma=0.00, clip=0.00):
        torch.manual_seed(713)
        self.args = args
        self.num_points = args.num_points
        self.dataset = dataset
        list_order = torch.randperm(len(self.dataset))
        self.sigma = sigma
        self.clip = clip

    def __len__(self):
        return len(self.dataset)

    def random_transform(self, p0):

        R = torch.from_numpy(Rot.random().as_matrix()).to(dtype=torch.float32)
        t = torch.rand(3, 1)
        p1 = R @ p0.T + t
        p1 = p1.T

        igt = torch.eye(4)
        igt[:3, :3] = R
        igt[:3, 3:] = t

        return p1, igt

    def __getitem__(self, index):
        p0, _ = self.dataset[index]  # one point cloud
        # breakpoint()
        if p0.shape[0] > self.num_points:
            idx_range = torch.arange(0, p0.shape[0])
            idx_samples = torch.multinomial(idx_range.float(), self.num_points)
            p0_ = p0[idx_samples, :]
            del p0
        else:
            p0_ = p0
            del p0

        p_ = add_noise(p0_, sigma=self.sigma, clip=self.clip)
        p1, igt = self.random_transform(p_)

        # p0_: template, p1: source, igt:transform matrix from p0 to p1
        data = preprocess(p0_, p1, igt,
                          voxel=self.args.voxel,
                          max_voxel_points=self.args.max_voxel_points,
                          num_voxels=self.args.num_voxels)
        return data


# by Rajat Talak
def preprocess(p0_pre, p1_pre, igt, voxel, max_voxel_points, num_voxels):

    # breakpoint()
    p0_pre = p0_pre.numpy()
    p1_pre = p1_pre.numpy()
    igt = igt.numpy()

    p0_pre_mean = np.mean(p0_pre, 0)
    p1_pre_mean = np.mean(p1_pre, 0)
    p0_pre_ = p0_pre - p0_pre_mean
    p1_pre_ = p1_pre - p1_pre_mean

    # voxelization
    p0, p1, xmin, ymin, zmin, xmax, ymax, zmax, vx, vy, vz = find_voxel_overlaps(p0_pre_, p1_pre_,
                                                                                 voxel)  # constraints of P1 ^ P2, where contains roughly overlapped area

    p0 = p0 + p0_pre_mean
    p1 = p1 + p1_pre_mean
    xmin0 = xmin + p0_pre_mean[0]
    ymin0 = ymin + p0_pre_mean[1]
    zmin0 = zmin + p0_pre_mean[2]
    xmax0 = xmax + p0_pre_mean[0]
    ymax0 = ymax + p0_pre_mean[1]
    zmax0 = zmax + p0_pre_mean[2]

    xmin1 = xmin + p1_pre_mean[0]
    ymin1 = ymin + p1_pre_mean[1]
    zmin1 = zmin + p1_pre_mean[2]
    xmax1 = xmax + p1_pre_mean[0]
    ymax1 = ymax + p1_pre_mean[1]
    zmax1 = zmax + p1_pre_mean[2]

    voxels_p0, coords_p0, num_points_per_voxel_p0 = points_to_voxel_second(p0,
                                                                           (xmin0, ymin0, zmin0, xmax0, ymax0, zmax0),
                                                                           (vx, vy, vz),
                                                                           max_voxel_points,
                                                                           reverse_index=False,
                                                                           max_voxels=num_voxels)
    voxels_p1, coords_p1, num_points_per_voxel_p1 = points_to_voxel_second(p1,
                                                                           (xmin1, ymin1, zmin1, xmax1, ymax1, zmax1),
                                                                           (vx, vy, vz), max_voxel_points,
                                                                           reverse_index=False,
                                                                           max_voxels=num_voxels)

    coords_p0_idx = coords_p0[:, 1] * (int(voxel ** 2)) + coords_p0[:, 0] * (int(voxel)) + coords_p0[:, 2]
    coords_p1_idx = coords_p1[:, 1] * (int(voxel ** 2)) + coords_p1[:, 0] * (int(voxel)) + coords_p1[:, 2]

    # calculate for the voxel medium
    xm_x0 = np.linspace(xmin0 + vx / 2, xmax0 - vx / 2, int(voxel))
    xm_y0 = np.linspace(ymin0 + vy / 2, ymax0 - vy / 2, int(voxel))
    xm_z0 = np.linspace(zmin0 + vz / 2, zmax0 - vz / 2, int(voxel))
    mesh3d0 = np.vstack(np.meshgrid(xm_x0, xm_y0, xm_z0)).reshape(3, -1).T
    xm_x1 = np.linspace(xmin1 + vx / 2, xmax1 - vx / 2, int(voxel))
    xm_y1 = np.linspace(ymin1 + vy / 2, ymax1 - vy / 2, int(voxel))
    xm_z1 = np.linspace(zmin1 + vz / 2, zmax1 - vz / 2, int(voxel))
    mesh3d1 = np.vstack(np.meshgrid(xm_x1, xm_y1, xm_z1)).reshape(3, -1).T

    voxel_coords_p0 = mesh3d0[coords_p0_idx]
    voxel_coords_p1 = mesh3d1[coords_p1_idx]

    # find voxels where number of points >= 80% of the maximum number of points
    idx_conditioned_p0 = coords_p0_idx[np.where(num_points_per_voxel_p0 >= 0.1 * max_voxel_points)]
    idx_conditioned_p1 = coords_p1_idx[np.where(num_points_per_voxel_p1 >= 0.1 * max_voxel_points)]
    idx_conditioned, _, _ = np.intersect1d(idx_conditioned_p0, idx_conditioned_p1, assume_unique=True,
                                           return_indices=True)
    _, _, idx_p0 = np.intersect1d(idx_conditioned, coords_p0_idx, assume_unique=True, return_indices=True)
    _, _, idx_p1 = np.intersect1d(idx_conditioned, coords_p1_idx, assume_unique=True, return_indices=True)
    voxel_coords_p0 = voxel_coords_p0[idx_p0]
    voxel_coords_p1 = voxel_coords_p1[idx_p1]
    voxels_p0 = voxels_p0[idx_p0]
    voxels_p1 = voxels_p1[idx_p1]

    return voxels_p0, voxel_coords_p0, voxels_p1, voxel_coords_p1, igt, p0, p1


