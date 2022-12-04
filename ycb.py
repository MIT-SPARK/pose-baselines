import copy
import csv
import json
import numpy as np
import open3d as o3d
import os
import pytorch3d
import sys
import torch
from pytorch3d import transforms, ops

DATASET_PATH: str = '../../data/ycb/models/ycb/'
sys.path.append("../../")

from c3po.models.modelgen import ModelFromShape
from c3po.utils.general import pos_tensor_to_o3d
from c3po.utils.visualization_utils import visualize_model_n_keypoints, visualize_torch_model_n_keypoints
import c3po.utils.general as gu

MODEL_TO_KPT_GROUPS = {
    "003_cracker_box": [set([0,1,3,4]), set([0,1,2,5]), set([1,2,3,7]), set([0,2,3,6]), \
                        set([4,5,6,0]), set([4,5,7,1]), set([5,6,7,2]), set([4,6,7,3])],
    "004_sugar_box": [set([0,1,3,6]), set([0,1,2,7]), set([0,2,3,5]), set([1,2,3,4]), \
                      set([4,6,7,1]), set([6,7,5,0]), set([4,5,7,2]), set([4,5,6,3])],
    "008_pudding_box": [set([0,1,2,6]), set([0,1,3,7]), set([1,2,3,5]), set([0,2,3,4]), \
                        set([4,6,7,0]), set([5,6,7,1]), set([4,5,7,3]), set([4,5,6,2])],
    "009_gelatin_box": [set([0,1,3,6]), set([0,1,2,7]), set([1,2,3,5]), set([0,2,3,4]), \
                        set([4,6,7,0]), set([5,6,7,1]), set([4,5,7,2]), set([4,5,6,3])],
    "010_potted_meat_can": [set([1,2,3,7]), set([0,2,3,6]), set([0,1,3,5]), set([0,1,2,4]), \
                            set([4,6,7,2]), set([5,6,7,3]), set([4,5,6,0]), set([4,5,7,1])],
    "019_pitcher_base": [set([8]), set([10])],
    "035_power_drill": [set([9]), set([10]), set([11]), set([12])],
    "036_wood_block": [set([0,2,3,7]), set([1,2,3,6]), set([0,1,2,5]), set([0,1,3,4]), \
                        set([4,6,7,3]), set([5,6,7,2]), set([4,5,6,1]), set([4,5,7,0])],
    "061_foam_brick": [set([0,2,3,5]), set([1,2,3,4]), set([0,1,3,7]), set([0,1,2,6]), \
                        set([4,5,6,2]), set([4,5,7,3]), set([4,6,7,1]), set([5,6,7,0])]
    }

SYMMETRIC_MODEL_IDS = ["001_chips_can", "002_master_chef_can", "003_cracker_box", "004_sugar_box", \
                       "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box" \
                       "009_gelatin_box", "010_potted_meat_can", "036_wood_block", "040_large_marker", \
                       "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"]


def get_model_and_keypoints(model_id):
    """
    Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints from the KeypointNet dataset.

    inputs:
    model_id    : string

    output:
    mesh        : o3d.geometry.TriangleMesh
    pcd         : o3d.geometry.PointCloud
    keypoints   : o3d.utils.Vector3dVector(nx3)
    """

    object_mesh_file = DATASET_PATH + model_id + '/poisson/nontextured.ply'
    mesh = o3d.io.read_triangle_mesh(filename=object_mesh_file)
    mesh.compute_vertex_normals() #how long does this take
    pcd = None
    kpt_filename = os.path.join(DATASET_PATH + model_id, "kpts_xyz.npy")
    keypoints_xyz = np.load(kpt_filename)

    return mesh, pcd, keypoints_xyz


def visualize_model(model_id):
    """ Given class_id and model_id this function outputs the colored mesh and keypoints
    from the ycb dataset and plots them using open3d.visualization.draw_geometries"""

    mesh, _, keypoints_xyz = get_model_and_keypoints(model_id=model_id)

    keypoint_markers = visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)

    return mesh, None, keypoints_xyz, keypoint_markers


class SE3PointCloudYCB(torch.utils.data.Dataset):
    """
    Given model_id, and number of points generates various point clouds and SE3 transformations
    of the ycb object.

    Returns a batch of
        input_point_cloud, keypoints, rotation, translation
    """
    def __init__(self, model_id, num_of_points=1000, dataset_len=10000):
        """
        model_id        : str   : model id of a ycb object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """

        self.model_id = model_id
        self.num_of_points = num_of_points
        self.len = dataset_len

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(model_id)
        #center the cad model
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """

        R = transforms.random_rotation()
        t = torch.rand(3, 1)

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return R @ model_pcd_torch + t, R @ self.keypoints_xyz.squeeze(0) + t, R, t

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ycb model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_model_keypoints(self):
        """
        Returns keypoints of the ycb model annotated in the KeypointNet dataset.

        output:
        model_keypoints : torch.tensor of shape (1, 3, N)

        where
        N = number of keypoints
        """

        return self.keypoints_xyz

    def _get_diameter(self):
        """
        Returns the diameter of the mid-sized object.

        output  :   torch.tensor of shape (1)
        """

        return self.diameter

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0

class SE3PointCloudYCBAugment(torch.utils.data.Dataset):
    """
    Given model_id, and number of points generates various point clouds and SE3 transformations
    of the ycb object with all points perturbed by some gaussian noise.

    Returns a batch of
        input_point_cloud, keypoints, rotation, translation
    """
    def __init__(self, model_id, num_of_points=1000, dataset_len=10000):
        """
        model_id        : str   : model id of a ycb object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """

        self.model_id = model_id
        self.num_of_points = num_of_points
        self.len = dataset_len

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(model_id)
        #center the cad model
        center = self.model_mesh.get_center()
        self.model_mesh.translate(-center)

        self.keypoints_xyz = self.keypoints_xyz - center
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """

        R = transforms.random_rotation()
        t = torch.rand(3, 1)

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        new_points = np.asarray(model_pcd.points)
        random_noise = np.random.normal(0, .005, new_points.shape)
        new_points = new_points + random_noise

        model_pcd_torch = torch.from_numpy(new_points).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return R @ model_pcd_torch + t, R @ self.keypoints_xyz.squeeze(0) + t, R, t

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ycb model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_model_keypoints(self):
        """
        Returns keypoints of the ycb model annotated in the KeypointNet dataset.

        output:
        model_keypoints : torch.tensor of shape (1, 3, N)

        where
        N = number of keypoints
        """

        return self.keypoints_xyz

    def _get_diameter(self):
        """
        Returns the diameter of the mid-sized object.

        output  :   torch.tensor of shape (1)
        """

        return self.diameter

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0

class DepthYCB(torch.utils.data.Dataset):
    """
    Given model_id and split, get real depth images from YCB dataset.

    Returns a batch of
        input_point_cloud, keypoints, rotation, translation
    """
    def __init__(self, model_id, split='train', num_of_points=500, only_load_nondegenerate_pcds = False):
        """
        model_id        : str   : model id of a ycb object
        num_of_points   : int   : max. number of points the depth point cloud will contain

        """

        self.model_id = model_id
        self.split = split
        self.num_of_points = num_of_points

        self.pcd_data_root = os.path.join(DATASET_PATH + model_id, "clouds/largest_cluster/")
        self.split_filenames = np.load(self.pcd_data_root + split + '_split.npy')
        self.len = self.split_filenames.shape[0]

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(model_id)
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

        if only_load_nondegenerate_pcds:
            print("ONLY LOADING NONDEGENERATE VIEWPOINTS USING KEYPOINT GROUPS")
            self.filter_degenerate(MODEL_TO_KPT_GROUPS)

        print("dataset len", self.len)


    def filter_degenerate(self, model_to_groups):
        """
        :param model
        :param predicted_point_cloud:
        :param predicted_keypoints:
        :param model_to_groups: dictionary from model to sets of points the predicted_point_cloud needs to be within distance_threshold of
        :param distance_threshold:
        :return: True if input_point_cloud is_degenerate, False otherwise
        """

        if self.model_id not in model_to_groups:
            print("cannot determine degeneracies of this model")
        point_groups = model_to_groups[self.model_id]
        keep_idxs = []
        for idx in range(len(self.split_filenames)):
            filename = self.split_filenames[idx]
            is_degenerate = self.is_pcd_degenerate(filename, point_groups)
            if not is_degenerate:
                keep_idxs.append(idx)
        self.split_filenames = self.split_filenames[keep_idxs]
        self.len = self.split_filenames.shape[0]

        # return self.split_filenames

    def is_pcd_degenerate(self, pcd_filename, groups):
        """

        :param pcd_filename:
        :param groups: list of keypoint indices we want pointcloud points close to
        :return:
        """
        pcd = o3d.io.read_point_cloud(self.pcd_data_root + pcd_filename)
        pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
        pcd_torch = pcd_torch.to(torch.float)
        _, viewpoint_camera, reference_camera, viewpoint_angle, _ = tuple(pcd_filename.split('_'))
        rgbFromObj_filename = os.path.join(DATASET_PATH + self.model_id, "poses/gt_wrt_rgb/",
                                           '{0}_{1}_pose.npy'.format(viewpoint_camera, viewpoint_angle))
        rgbFromObj = np.load(rgbFromObj_filename)
        R_true = torch.from_numpy(rgbFromObj[:3, :3]).to(torch.float)
        t_true = torch.from_numpy(rgbFromObj[:3, 3]).unsqueeze(-1).to(torch.float)

        transformed_keypoints_gt = R_true @ self.keypoints_xyz.squeeze(0) + t_true
        for group in groups:
            kpt_group = transformed_keypoints_gt[:,list(group)]
            if self.is_close(kpt_group, pcd_torch):
                return False
        return True

    def is_close(self, keypoints, pcd, threshold=0.015):
        pcd = pcd.unsqueeze(0)
        keypoints = keypoints.unsqueeze(0)
        closest_dist, _, _ = ops.knn_points(torch.transpose(keypoints, -1, -2), torch.transpose(pcd, -1, -2), K=1,
                                          return_sorted=False)
        if torch.max(torch.sqrt(closest_dist)) < threshold:
            return True
        return False

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """
        pcd = o3d.io.read_point_cloud(self.pcd_data_root + self.split_filenames[idx])
        pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
        pcd_torch = pcd_torch.to(torch.float)

        #downsample to number of points expected
        m = pcd_torch.shape[-1]
        if m > self.num_of_points:
            shuffle_idxs = torch.randperm(m)
            point_cloud = pcd_torch[:, shuffle_idxs[:self.num_of_points]]
        elif m < self.num_of_points:
            #pad with zeros
            pc_pad = torch.zeros(3, self.num_of_points - m) #how many additional points to add
            point_cloud = torch.cat([pcd_torch, pc_pad], dim=1)
        else:
            point_cloud = pcd_torch

        #load ground truth R, ground truth t
        _, viewpoint_camera, reference_camera, viewpoint_angle, _ = tuple(self.split_filenames[idx].split('_'))
        # return R @ model_pcd_torch + t, R, t
        rgbFromObj_filename = os.path.join(DATASET_PATH + self.model_id, "poses/gt_wrt_rgb/",
                                           '{0}_{1}_pose.npy'.format(viewpoint_camera, viewpoint_angle))
        rgbFromObj = np.load(rgbFromObj_filename)
        R_true = torch.from_numpy(rgbFromObj[:3, :3]).to(torch.float)
        t_true = torch.from_numpy(rgbFromObj[:3,3]).unsqueeze(-1).to(torch.float)

        return point_cloud, R_true @ self.keypoints_xyz.squeeze(0) + t_true, R_true, t_true

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ycb model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_model_keypoints(self):
        """
        Returns keypoints of the ycb model annotated in the KeypointNet dataset.

        output:
        model_keypoints : torch.tensor of shape (1, 3, N)

        where
        N = number of keypoints
        """

        return self.keypoints_xyz

    def _get_diameter(self):
        """
        Returns the diameter of the mid-sized object.

        output  :   torch.tensor of shape (1)
        """

        return self.diameter

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0

class DepthYCBAugment(torch.utils.data.Dataset):
    """
    Given model_id and split, get real depth images from YCB dataset.

    Returns a batch of
        input_point_cloud, keypoints, rotation, translation
    """
    def __init__(self, model_id, split='train', num_of_points=500, only_load_nondegenerate_pcds = False):
        """
        model_id        : str   : model id of a ycb object
        num_of_points   : int   : max. number of points the depth point cloud will contain

        """

        self.model_id = model_id
        self.split = split
        self.num_of_points = num_of_points

        self.pcd_data_root = os.path.join(DATASET_PATH + model_id, "clouds/data_augmentation/")
        if only_load_nondegenerate_pcds and os.path.exists(self.pcd_data_root + split + '_split_wo_degeneracy.npy'):
            print("ONLY LOADING NONDEGENERATE VIEWPOINTS")
            self.split_filenames = np.load(self.pcd_data_root + split + '_split_wo_degeneracy.npy')
        else:
            self.split_filenames = np.load(self.pcd_data_root + split + '_split.npy')
        self.len = self.split_filenames.shape[0]
        print("dataset len", self.len)

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(model_id)
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))


    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """
        pcd = o3d.io.read_point_cloud(self.pcd_data_root + self.split_filenames[idx])
        pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
        pcd_torch = pcd_torch.to(torch.float)

        #downsample to number of points expected
        m = pcd_torch.shape[-1]
        if m > self.num_of_points:
            shuffle_idxs = torch.randperm(m)
            point_cloud = pcd_torch[:, shuffle_idxs[:self.num_of_points]]
        elif m < self.num_of_points:
            #pad with zeros
            pc_pad = torch.zeros(3, self.num_of_points - m) #how many additional points to add
            point_cloud = torch.cat([pcd_torch, pc_pad], dim=1)
        else:
            point_cloud = pcd_torch

        #load ground truth R, ground truth t
        _, viewpoint_camera, reference_camera, viewpoint_angle, _, _, _ = tuple(self.split_filenames[idx].split('_'))
        rgbFromObj_filename = os.path.join(DATASET_PATH + self.model_id, "poses/gt_wrt_rgb/",
                                           '{0}_{1}_pose.npy'.format(viewpoint_camera, viewpoint_angle))
        rgbFromObj = np.load(rgbFromObj_filename)
        R_true = torch.from_numpy(rgbFromObj[:3, :3]).to(torch.float)
        t_true = torch.from_numpy(rgbFromObj[:3,3]).unsqueeze(-1).to(torch.float)

        return point_cloud, R_true @ self.keypoints_xyz.squeeze(0) + t_true, R_true, t_true

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ycb model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_model_keypoints(self):
        """
        Returns keypoints of the ycb model annotated in the KeypointNet dataset.

        output:
        model_keypoints : torch.tensor of shape (1, 3, N)

        where
        N = number of keypoints
        """

        return self.keypoints_xyz

    def _get_diameter(self):
        """
        Returns the diameter of the mid-sized object.

        output  :   torch.tensor of shape (1)
        """

        return self.diameter

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0

class MixedDepthYCBAugment(torch.utils.data.Dataset):
    """
    Given model_id and split, get real depth images from YCB dataset.
    Dataset is a total of 5000 images with 1000 each of 002, 006, 011, 037, 052

    Returns a batch of
        input_point_cloud, rotation, translation
    """
    def __init__(self, model_id, split='train', num_of_points=500, mixed_data=True):
        """
        model_id        : str   : model id of a ycb object
        num_of_points   : int   : max. number of points the depth point cloud will contain

        """

        self.model_id = model_id
        self.split = split
        self.num_of_points = num_of_points

        if mixed_data and split != 'test':
            self.pcd_data_root = os.path.join(DATASET_PATH + "/mixed/")
        if split == 'test':
            self.pcd_data_root = os.path.join(DATASET_PATH + model_id, "clouds/largest_cluster/")
        else:
            self.pcd_data_root = os.path.join(DATASET_PATH + model_id, "clouds/data_augmentation/")
        self.split_filenames = np.load(self.pcd_data_root + split + '_split.npy')
        self.len = self.split_filenames.shape[0]
        print("dataset len", self.len)

        # get model
        self.model_mesh, _, self.keypoints_xyz = get_model_and_keypoints(model_id)
        self.keypoints_xyz = torch.from_numpy(self.keypoints_xyz).transpose(0, 1).unsqueeze(0).to(torch.float)

        # size of the model
        self.diameter = np.linalg.norm(np.asarray(self.model_mesh.get_max_bound()) - np.asarray(self.model_mesh.get_min_bound()))

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """
        pcd = o3d.io.read_point_cloud(self.pcd_data_root + self.split_filenames[idx])
        pcd_torch = torch.from_numpy(np.asarray(pcd.points)).transpose(0, 1)  # (3, m)
        pcd_torch = pcd_torch.to(torch.float)

        #downsample to number of points expected
        m = pcd_torch.shape[-1]
        if m > self.num_of_points:
            shuffle_idxs = torch.randperm(m)
            point_cloud = pcd_torch[:, shuffle_idxs[:self.num_of_points]]
        elif m < self.num_of_points:
            #pad with zeros
            pc_pad = torch.zeros(3, self.num_of_points - m) #how many additional points to add
            point_cloud = torch.cat([pcd_torch, pc_pad], dim=1)
        else:
            point_cloud = pcd_torch

        #load ground truth R, ground truth t
        full_pcd_filename = tuple(self.split_filenames[idx].split("/"))
        if len(full_pcd_filename) == 2:
            _, pcd_filename = full_pcd_filename
        else:
            pcd_filename = full_pcd_filename[0]

        _, viewpoint_camera, reference_camera, viewpoint_angle, *_ = tuple(self.split_filenames[idx].split('_'))

        rgbFromObj_filename = os.path.join(DATASET_PATH + self.model_id, "poses/gt_wrt_rgb/",
                                           '{0}_{1}_pose.npy'.format(viewpoint_camera, viewpoint_angle))
        rgbFromObj = np.load(rgbFromObj_filename)
        R_true = torch.from_numpy(rgbFromObj[:3, :3]).to(torch.float)
        t_true = torch.from_numpy(rgbFromObj[:3,3]).unsqueeze(-1).to(torch.float)

        return point_cloud, R_true, t_true

    def _get_cad_models(self):
        """
        Returns a sampled point cloud of the ycb model with self.num_of_points points.

        output:
        cad_models  : torch.tensor of shape (1, 3, self.num_of_points)

        """

        model_pcd = self.model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        model_pcd_torch = torch.from_numpy(np.asarray(model_pcd.points)).transpose(0, 1)  # (3, m)
        model_pcd_torch = model_pcd_torch.to(torch.float)

        return model_pcd_torch.unsqueeze(0)

    def _get_model_keypoints(self):
        """
        Returns keypoints of the ycb model annotated in the KeypointNet dataset.

        output:
        model_keypoints : torch.tensor of shape (1, 3, N)

        where
        N = number of keypoints
        """

        return self.keypoints_xyz

    def _get_diameter(self):
        """
        Returns the diameter of the mid-sized object.

        output  :   torch.tensor of shape (1)
        """

        return self.diameter

    def _visualize(self):
        """
        Visualizes the two CAD models and the corresponding keypoints

        """

        cad_models = self._get_cad_models()
        model_keypoints = self._get_model_keypoints()
        visualize_torch_model_n_keypoints(cad_models=cad_models, model_keypoints=model_keypoints)

        return 0


if __name__ == "__main__":

    model_id = "021_bleach_cleanser"  # a particular chair model
    batch_size = 5
    #
    #
    print("Test: DepthYCB()")
    dataset = DepthYCB(model_id=model_id, split='test')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(loader):
        pc, kp, R, t = data
        print(pc.shape)
        print(kp.shape)
        print(R.shape)
        print(t.shape)
        visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 5:
            break

    print("Test: DepthYCBAugment")
    dataset = DepthYCBAugment(model_id=model_id, split='test')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(loader):
        pc, kp, R, t = data
        print(pc.shape)
        print(kp.shape)
        print(R.shape)
        print(t.shape)
        visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 5:
            break

    print("Test: SE3PointCloudYCB()")
    dataset = SE3PointCloudYCB(model_id=model_id)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(loader):
        pc, kp, R, t = data
        print(pc.shape)
        print(kp.shape)
        print(R.shape)
        print(t.shape)
        visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 5:
            break

    print("Test: SE3PointCloudYCBAugment")
    dataset = SE3PointCloudYCBAugment(model_id=model_id)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(loader):
        pc, kp, R, t = data
        print(pc.shape)
        print(kp.shape)
        print(R.shape)
        print(t.shape)
        visualize_torch_model_n_keypoints(cad_models=pc, model_keypoints=kp)
        if i >= 5:
            break

    print("Test: MixedDepthYCBAugment")
    dataset = MixedDepthYCBAugment(model_id=model_id, split='test')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(loader):
        pc, R, t = data
        print(pc.shape)
        print(R.shape)
        print(t.shape)
        if i >= 5:
            break

    print("Test: get_model_and_keypoints()")
    mesh, _, keypoints_xyz = get_model_and_keypoints(model_id=model_id)

    print("Test: visualize_model_n_keypoints()")
    visualize_model_n_keypoints([mesh], keypoints_xyz=keypoints_xyz)

    print("Test: visualize_model()")
    visualize_model(model_id=model_id)
