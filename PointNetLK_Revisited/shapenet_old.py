import numpy as np
import yaml
import json
import torch
import torchvision
import open3d as o3d
from pathlib import Path
import copy
from scipy.spatial.transform import Rotation as Rot
# from pytorch3d import transforms, ops
import random

import data_utils
import modelnet

# BASE_DIR = Path(__file__).parent.parent

# expt_shapenet_dir = Path(__file__).parent.parent.parent.parent / 'expt_shapenet'
ANNOTATIONS_FOLDER: str = './data_shapenet/KeypointNet/KeypointNet/annotations/'
PCD_FOLDER_NAME: str = './data_shapenet/KeypointNet/KeypointNet/pcds/'
MESH_FOLDER_NAME: str = './data_shapenet/KeypointNet/ShapeNetCore.v2.ply/'
OBJECT_CATEGORIES: list = ['airplane', 'bathtub', 'bed', 'bottle',
                           'cap', 'car', 'chair', 'guitar',
                           'helmet', 'knife', 'laptop', 'motorcycle',
                           'mug', 'skateboard', 'table', 'vessel']
CLASS_ID: dict = {'airplane': "02691156",
                  'bathtub': "02808440",
                  'bed': "02818832",
                  'bottle': "02876657",
                  'cap': "02954340",
                  'car': "02958343",
                  'chair': "03001627",
                  'guitar': "03467517",
                  'helmet': "03513137",
                  'knife': "03624134",
                  'laptop': "03642806",
                  'motorcycle': "03790512",
                  'mug': "03797390",
                  'skateboard': "04225987",
                  'table': "04379243",
                  'vessel': "04530566"}

CLASS_NAME: dict = {"02691156": 'airplane',
                    "02808440": 'bathtub',
                    "02818832": 'bed',
                    "02876657": 'bottle',
                    "02954340": 'cap',
                    "02958343": 'car',
                    "03001627": 'chair',
                    "03467517": 'guitar',
                    "03513137": 'helmet',
                    "03624134": 'knife',
                    "03642806": 'laptop',
                    "03790512": 'motorcycle',
                    "03797390": 'mug',
                    "04225987": 'skateboard',
                    "04379243": 'table',
                    "04530566": 'vessel'}

CLASS_MODEL_ID: dict = {'airplane': '3db61220251b3c9de719b5362fe06bbb',
                        'bathtub': '90b6e958b359c1592ad490d4d7fae486',
                        'bed': '7c8eb4ab1f2c8bfa2fb46fb8b9b1ac9f',
                        'bottle': '41a2005b595ae783be1868124d5ddbcb',
                        'cap': '3dec0d851cba045fbf444790f25ea3db',
                        'car': 'ad45b2d40c7801ef2074a73831d8a3a2',
                        'chair': '1cc6f2ed3d684fa245f213b8994b4a04',
                        'guitar': '5df08ba7af60e7bfe72db292d4e13056',
                        'helmet': '3621cf047be0d1ae52fafb0cab311e6a',
                        'knife': '819e16fd120732f4609e2d916fa0da27',
                        'laptop': '519e98268bee56dddbb1de10c9529bf7',
                        'motorcycle': '481f7a57a12517e0fe1b9fad6c90c7bf',
                        'mug': 'f3a7f8198cc50c225f5e789acd4d1122',
                        'skateboard': '98222a1e5f59f2098745e78dbc45802e',
                        'table': '3f5daa8fe93b68fa87e2d08958d6900c',
                        'vessel': '5c54100c798dd681bfeb646a8eadb57'}


def get_radius(object_diameter, cam_location):
    """ returns radius, which is the maximum distance from cam_location within which all points in the object lie"""
    return 100*np.sqrt(object_diameter**2 + np.linalg.norm(cam_location)**2)


def get_depth_pcd(centered_pcd, camera, radius, method='1'):
    """ This produces a depth point cloud. Input:
    centered_pcd (o3d.geometry.PointCloud object) = pcd that is centered at (0, 0, 0)
    camera (numpy.ndarray[float64[3, 1]])         = location of camera in the 3d space
    radius (float)                                = radius from camera location, beyond which points are not taken
    """
    pcd = copy.deepcopy(centered_pcd)

    """Method 1"""
    if method == '1':
        _, pt_map = pcd.hidden_point_removal(camera_location=camera, radius=radius)
        pcd = pcd.select_by_index(pt_map)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        return pcd

    """Method 2"""
    # Do not use Method 2. It constructs an artificial mesh from sampled/visible points.
    # This leads it to connect points that belong to distinct objects, thereby changing things.
    if method == '2':
        visible_mesh, _ = pcd.hidden_point_removal(camera_location=camera, radius=radius)
        pcd_visible = visible_mesh.sample_points_uniformly(number_of_points=10000)
        pcd_visible.paint_uniform_color([0.5, 0.5, 0.5])

        return pcd_visible


def get_model_and_keypoints(class_id, model_id):
    """
    Given class_id and model_id this function outputs the colored mesh, pcd, and keypoints from the KeypointNet dataset.

    inputs:
    class_id    : string
    model_id    : string

    output:
    mesh        : o3d.geometry.TriangleMesh
    pcd         : o3d.geometry.PointCloud
    keypoints   : o3d.utils.Vector3dVector(nx3)
    """

    object_pcd_file = PCD_FOLDER_NAME + str(class_id) + '/' + str(model_id) + '.pcd'
    object_mesh_file = MESH_FOLDER_NAME + str(class_id) + '/' + str(model_id) + '.ply'

    pcd = o3d.io.read_point_cloud(filename=object_pcd_file)
    mesh = o3d.io.read_triangle_mesh(filename=object_mesh_file)
    mesh.compute_vertex_normals()

    annotation_file = ANNOTATIONS_FOLDER + CLASS_NAME[str(class_id)] + '.json'
    file_temp = open(str(annotation_file))
    anotation_data = json.load(file_temp)

    for idx, entry in enumerate(anotation_data):
        if entry['model_id'] == str(model_id):
            keypoints = entry['keypoints']
            break

    keypoints_xyz = []
    for aPoint in keypoints:
        keypoints_xyz.append(aPoint['xyz'])

    keypoints_xyz = np.array(keypoints_xyz)

    return mesh, pcd, keypoints_xyz


class SE3PointCloudAll(torch.utils.data.Dataset):
    """
    Randomly generates ShapeNet object point clouds and their SE3 transformations.

    Returns a batch of
        pc1, pc2, kp1, kp2, rotation, translation

    """
    def __init__(self, num_of_points=1024, dataset_len=2048, transform=None,
                 dir_location='../../data/learning-objects/keypointnet_datasets/'):
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """
        self.objects = OBJECT_CATEGORIES
        self.num_of_points = num_of_points
        self.len = dataset_len

        self.transform = transform

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """

        # randomly choose an object category name
        class_name = random.choice(self.objects)
        class_id = CLASS_ID[class_name]
        model_id = CLASS_MODEL_ID[class_name]

        model_mesh, _, kp = get_model_and_keypoints(class_id, model_id)
        center = model_mesh.get_center()
        model_mesh.translate(-center)
        kp = kp - center
        kp1 = torch.from_numpy(kp).transpose(0, 1).unsqueeze(0).to(torch.float)

        # diameter = np.linalg.norm(np.asarray(model_mesh.get_max_bound()) - np.asarray(model_mesh.get_min_bound()))
        R = torch.from_numpy(Rot.random().as_matrix()).to(dtype=torch.float32)
        t = torch.rand(3, 1)

        pc1_pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        pc1 = torch.from_numpy(np.asarray(pc1_pcd.points)).transpose(0, 1)
        pc1 = pc1.to(torch.float)

        # Apply random transform
        if self.transform is not None:
            # breakpoint()
            pc1 = self.transform(pc1.T).T


        pc2 = R @ pc1 + t
        kp2 = R @ kp1 + t

        # T = torch.eye(4)
        # T[:3, :3] = R
        # T[:3, 3:] = t
        return (pc1, pc2, kp1, kp2, R, t)
        # return pc1.T, pc2.T, T


class DepthPCAll(torch.utils.data.Dataset):
    """
    Randomly generates ShapeNet object point clouds and their SE3 transformed depth rendering.
    pc2 is depth point cloud.
    This doesn't do zero padding for depth point clouds.

    Returns a batch of
        pc1, pc2, kp1, kp2, rotation, translation

    """
    def __init__(self, num_of_points1=1024, radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points2=2048, dataset_len=10000, rotate_about_z=True, transform=None):
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """
        self.objects = OBJECT_CATEGORIES
        self.num_of_points_pc1 = num_of_points1
        self.len = dataset_len
        self.num_of_points_pc2 = num_of_points2
        self.radius_multiple = radius_multiple
        self.rotate_about_z = rotate_about_z
        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1)
        self.pi = float(np.pi)
        self.transform = transform
        # set a camera location, with respect to the origin

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """

        # randomly choose an object category name
        class_name = random.choice(self.objects)
        class_id = CLASS_ID[class_name]
        model_id = CLASS_MODEL_ID[class_name]

        model_mesh, _, kp = get_model_and_keypoints(class_id, model_id)
        center = model_mesh.get_center()
        model_mesh.translate(-center)
        kp = kp - center

        # computing diameter
        diameter = np.linalg.norm(np.asarray(model_mesh.get_max_bound()) - np.asarray(model_mesh.get_min_bound()))

        # extracting the first data
        kp1 = torch.from_numpy(kp).transpose(0, 1).unsqueeze(0).to(torch.float)
        pc1_pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_pc1)
        pc1 = torch.from_numpy(np.asarray(pc1_pcd.points)).transpose(0, 1)  # (3, m)
        pc1 = pc1.to(torch.float)

        # apply transform
        if self.transform is not None:
            pc1 = self.transform(pc1.T).T

        # apply random rotation
        if self.rotate_about_z:
            R = torch.eye(3)
            angle = 2 * self.pi * torch.rand(1)
            c = torch.cos(angle)
            s = torch.sin(angle)

            # # z
            # R[0, 0] = c
            # R[0, 1] = -s
            # R[1, 0] = s
            # R[1, 1] = c

            # # x
            # R[1, 1] = c
            # R[1, 2] = -s
            # R[2, 1] = s
            # R[2, 2] = c

            # y
            R[0, 0] = c
            R[0, 2] = s
            R[2, 0] = -s
            R[2, 2] = c

        else:
            # R = transforms.random_rotation()
            R = torch.from_numpy(Rot.random().as_matrix()).to(dtype=torch.float32)

        model_mesh = model_mesh.rotate(R=R.numpy())

        # sample a point cloud from the self.model_mesh
        pc2_pcd_ = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_pc2)

        # take a depth image from a distance of the rotated self.model_mesh from self.camera_location
        beta = torch.rand(1, 1)
        camera_location_factor = beta * (self.radius_multiple[1] - self.radius_multiple[0]) + self.radius_multiple[0]
        camera_location_factor = camera_location_factor * diameter
        radius = get_radius(cam_location=camera_location_factor * self.camera_location.numpy(),
                               object_diameter=diameter)
        pc2_pcd = get_depth_pcd(centered_pcd=pc2_pcd_, camera=self.camera_location.numpy(), radius=radius)

        pc2 = torch.from_numpy(np.asarray(pc2_pcd.points)).transpose(0, 1)  # (3, m)
        pc2 = pc2.to(torch.float)

        # Translate by a random t
        t = torch.rand(3, 1)
        pc2 = pc2 + t
        kp2 = R @ kp1 + t

        # T = torch.eye(4)
        # T[:3, :3] = R
        # T[:3, 3:] = t
        return (pc1, pc2, kp1, kp2, R, t)
        # return pc1.T, pc2.T, T


class ShapeNetDataset(torch.utils.data.Dataset):
  def __init__(self,
               num_points,
               type='sim',
               dataset_length=2048,    # tunable parameter
               manual_seed=False,
               transform=None
               ):

    self.type = type
    self.n_points = num_points
    self.dataset_length = dataset_length
    self.transform = transform

    if self.type == 'real':
        self.ds = DepthPCAll(dataset_len=self.dataset_length,
                             num_of_points1=self.n_points,
                             num_of_points2=10*self.n_points,
                             transform=self.transform)
    elif self.type == 'sim':
        self.ds = SE3PointCloudAll(dataset_len=self.dataset_length,
                                   num_of_points=self.n_points,
                                   transform=self.transform)

    self.randg = np.random.RandomState()
    if manual_seed:
        self.reset_seed()

  def reset_seed(self, seed=0):
    # logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def __len__(self):

      return self.dataset_length

  def __getitem__(self, idx):

    if self.type == 'sim':
        pc0, pc1, _, _, R, t = self.ds[idx]
    elif self.type == 'real':
        pc0, pc1, _, _, R, t = self.ds[idx]
        while pc0.shape[1] > pc1.shape[1]:
            pc0, pc1, _, _, R, t = self.ds[idx]

        if pc0.shape[1] < pc1.shape[1]:
            idx_range = torch.arange(pc1.shape[1]).to(dtype=torch.float64)
            try:
                random_idx = torch.multinomial(idx_range, pc0.shape[1], replacement=False)
            except RuntimeError:
                breakpoint()
            pc1 = pc1[:, random_idx]

    xyz0 = pc0.T.numpy()
    xyz1 = pc1.T.numpy()

    xyz0 = np.ascontiguousarray(xyz0)
    xyz1 = np.ascontiguousarray(xyz1)

    # breakpoint()
    trans = torch.eye(4)
    trans[:3, :3] = R
    trans[:3, 3:] = t
    trans = trans.numpy()

    xyz0 = torch.from_numpy(xyz0)
    xyz1 = torch.from_numpy(xyz1)
    trans = torch.from_numpy(trans)

    # data = modelnet.preprocess(xyz0, xyz1, trans,
    #                            voxel=self.args.voxel,
    #                            max_voxel_points=self.args.max_voxel_points,
    #                            num_voxels=self.args.num_voxels)

    return xyz0, xyz1, trans


class SE3PointCloud(torch.utils.data.Dataset):
    """
    Randomly generates ShapeNet object point clouds and their SE3 transformations.

    Returns a batch of
        pc1, pc2, kp1, kp2, rotation, translation

    """
    def __init__(self, class_name, num_of_points=1024, dataset_len=2048, transform=None,
                 dir_location='../../data/learning-objects/keypointnet_datasets/'):
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """
        self.class_name = class_name
        self.objects = OBJECT_CATEGORIES
        self.transform = transform

        if self.class_name not in self.objects:
            raise ValueError("Specified class name is not correct.")

        self.num_of_points = num_of_points
        self.len = dataset_len

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """

        # randomly choose an object category name
        class_name = self.class_name
        class_id = CLASS_ID[class_name]
        model_id = CLASS_MODEL_ID[class_name]

        model_mesh, _, kp = get_model_and_keypoints(class_id, model_id)
        center = model_mesh.get_center()
        model_mesh.translate(-center)
        kp = kp - center
        kp1 = torch.from_numpy(kp).transpose(0, 1).unsqueeze(0).to(torch.float)

        # diameter = np.linalg.norm(np.asarray(model_mesh.get_max_bound()) - np.asarray(model_mesh.get_min_bound()))

        # R = transforms.random_rotation()  #TODO: replace pytorch3d to numpy function
        #TODO: R = get_random_rotation(max_angle=0.7)
        R = torch.from_numpy(Rot.random().as_matrix()).to(dtype=torch.float32)
        t = torch.rand(3, 1)

        pc1_pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        pc1 = torch.from_numpy(np.asarray(pc1_pcd.points)).transpose(0, 1)
        pc1 = pc1.to(torch.float)

        # apply transform
        if self.transform is not None:
            pc1 = self.transform(pc1.T).T

        pc2 = R @ pc1 + t
        kp2 = R @ kp1 + t

        return (pc1, pc2, kp1, kp2, R, t)


class DepthPC(torch.utils.data.Dataset):
    """
    Randomly generates ShapeNet object point clouds and their SE3 transformed depth rendering.
    pc2 is depth point cloud.
    This doesn't do zero padding for depth point clouds.

    Returns a batch of
        pc1, pc2, kp1, kp2, rotation, translation

    """
    def __init__(self, class_name, num_of_points1=1024, radius_multiple=torch.tensor([1.2, 3.0]),
                 num_of_points2=2048, dataset_len=10000, rotate_about_z=False, transform=None):
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """
        self.class_name = class_name
        self.objects = OBJECT_CATEGORIES
        if self.class_name not in self.objects:
            raise ValueError("Specified class name is not correct.")

        self.num_of_points_pc1 = num_of_points1
        self.len = dataset_len
        self.num_of_points_pc2 = num_of_points2
        self.radius_multiple = radius_multiple
        self.rotate_about_z = rotate_about_z
        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1)
        self.transform = transform
        # set a camera location, with respect to the origin

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """
        # randomly choose an object category name
        class_name = self.class_name
        class_id = CLASS_ID[class_name]
        model_id = CLASS_MODEL_ID[class_name]

        model_mesh, _, kp = get_model_and_keypoints(class_id, model_id)
        center = model_mesh.get_center()
        model_mesh.translate(-center)
        kp = kp - center

        # computing diameter
        diameter = np.linalg.norm(np.asarray(model_mesh.get_max_bound()) - np.asarray(model_mesh.get_min_bound()))

        # extracting the first data
        kp1 = torch.from_numpy(kp).transpose(0, 1).unsqueeze(0).to(torch.float)
        pc1_pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_pc1)
        pc1 = torch.from_numpy(np.asarray(pc1_pcd.points)).transpose(0, 1)  # (3, m)
        pc1 = pc1.to(torch.float)

        # apply transform
        if self.transform is not None:
            pc1 = self.transform(pc1.T).T

        # apply random rotation
        if self.rotate_about_z:
            R = torch.eye(3)
            angle = 2 * self.pi * torch.rand(1)
            c = torch.cos(angle)
            s = torch.sin(angle)

            # # z
            # R[0, 0] = c
            # R[0, 1] = -s
            # R[1, 0] = s
            # R[1, 1] = c

            # # x
            # R[1, 1] = c
            # R[1, 2] = -s
            # R[2, 1] = s
            # R[2, 2] = c

            # y
            R[0, 0] = c
            R[0, 2] = s
            R[2, 0] = -s
            R[2, 2] = c

        else:
            # R = transforms.random_rotation()
            R = torch.from_numpy(Rot.random().as_matrix()).to(dtype=torch.float32)

        model_mesh = model_mesh.rotate(R=R.numpy())

        # sample a point cloud from the self.model_mesh
        pc2_pcd_ = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points_pc2)

        # take a depth image from a distance of the rotated self.model_mesh from self.camera_location
        beta = torch.rand(1, 1)
        camera_location_factor = beta * (self.radius_multiple[1] - self.radius_multiple[0]) + self.radius_multiple[0]
        camera_location_factor = camera_location_factor * diameter
        radius = get_radius(cam_location=camera_location_factor * self.camera_location.numpy(),
                               object_diameter=diameter)
        pc2_pcd = get_depth_pcd(centered_pcd=pc2_pcd_, camera=self.camera_location.numpy(), radius=radius)

        pc2 = torch.from_numpy(np.asarray(pc2_pcd.points)).transpose(0, 1)  # (3, m)
        pc2 = pc2.to(torch.float)

        # Translate by a random t
        t = torch.rand(3, 1)
        pc2 = pc2 + t
        kp2 = R @ kp1 + t

        return (pc1, pc2, kp1, kp2, R, t)


class ShapeNetObjectDataset(torch.utils.data.Dataset):
  def __init__(self,
               object,
               num_points,
               type='sim',
               dataset_length=2048,    # tunable parameter
               manual_seed=False,
               transform=None
               ):
    self.type = type
    self.class_name = object
    self.n_points = num_points
    self.dataset_length = dataset_length
    self.transform = transform

    if self.type == 'real':
        self.ds = DepthPC(class_name=self.class_name,
                          dataset_len=self.dataset_length,
                          num_of_points1=self.n_points,
                          num_of_points2=10*self.n_points,
                          transform=self.transform)
    elif self.type == 'sim':
        self.ds = SE3PointCloud(class_name=self.class_name,
                                dataset_len=self.dataset_length,
                                num_of_points=self.n_points,
                                transform=self.transform)

    self.randg = np.random.RandomState()
    if manual_seed:
        self.reset_seed()

  def reset_seed(self, seed=0):
    # logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def __len__(self):

      return self.dataset_length

  def __getitem__(self, idx):

    if self.type == 'sim':
        pc0, pc1, _, _, R, t = self.ds[idx]
    elif self.type == 'real':
        pc0, pc1, _, _, R, t = self.ds[idx]
        while pc0.shape[1] > pc1.shape[1]:
            pc0, pc1, _, _, R, t = self.ds[idx]

        if pc0.shape[1] < pc1.shape[1]:
            idx_range = torch.arange(pc1.shape[1]).to(dtype=torch.float64)
            try:
                random_idx = torch.multinomial(idx_range, pc0.shape[1], replacement=False)
            except RuntimeError:
                breakpoint()
            pc1 = pc1[:, random_idx]

    xyz0 = pc0.T.numpy()
    xyz1 = pc1.T.numpy()

    xyz0 = np.ascontiguousarray(xyz0)
    xyz1 = np.ascontiguousarray(xyz1)

    # breakpoint()
    trans = torch.eye(4)
    trans[:3, :3] = R
    trans[:3, 3:] = t
    # trans = trans.numpy()

    # R = torch.from_numpy(R)
    # t = torch.from_numpy(t)
    xyz0 = torch.from_numpy(xyz0)
    xyz1 = torch.from_numpy(xyz1)
    # trans = torch.from_numpy(trans)

    # breakpoint()
    # data = modelnet.preprocess(xyz0, xyz1, trans,
    #                            voxel=self.args.voxel,
    #                            max_voxel_points=self.args.max_voxel_points,
    #                            num_voxels=self.args.num_voxels)

    return xyz0, xyz1, trans
    # return (xyz0.T, xyz1.T, None, None, R, t)


class ShapeNet(torch.utils.data.Dataset):
    """
    Randomly generates ShapeNet object point clouds and their SE3 transformed depth rendering.
    pc2 is depth point cloud.
    This doesn't do zero padding for depth point clouds.

    Returns a batch of
        pc1, pc2, kp1, kp2, rotation, translation

    """
    def __init__(self, object, num_of_points=1000, type='sim',
                 radius_multiple=torch.tensor([1.2, 3.0]),
                 dataset_len=10000, rotate_about_z=False):
        """
        class_id        : str   : class id of a ShapeNetCore object
        model_id        : str   : model id of a ShapeNetCore object
        num_of_points   : int   : max. number of points the depth point cloud will contain
        dataset_len     : int   : size of the dataset

        """
        self.class_name = object            #can be 'all'
        self.objects = OBJECT_CATEGORIES

        self.type = type
        # if self.class_name not in self.objects:
        #     raise ValueError("Specified class name is not correct.")

        self.num_of_points = num_of_points
        self.len = dataset_len
        self.radius_multiple = radius_multiple
        self.rotate_about_z = rotate_about_z
        self.camera_location = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(-1)
        # set a camera location, with respect to the origin

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        """
        output:
        point_cloud         : torch.tensor of shape (3, m)                  : the SE3 transformed point cloud
        R                   : torch.tensor of shape (3, 3)                  : rotation
        t                   : torch.tensor of shape (3, 1)                  : translation
        """
        # randomly choose an object category name
        if self.class_name == 'all':
            class_name = random.choice(self.objects)
        else:
            class_name = self.class_name
        class_id = CLASS_ID[class_name]
        model_id = CLASS_MODEL_ID[class_name]

        model_mesh, _, _ = get_model_and_keypoints(class_id, model_id)
        center = model_mesh.get_center()
        model_mesh.translate(-center)

        if self.type == 'sim':
            pc_pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
            pc0 = torch.from_numpy(np.asarray(pc_pcd.points)).transpose(0, 1)  # (3, m)
            pc0 = pc0.to(torch.float)
            pc = pc0
            del pc_pcd, model_mesh

        elif self.type == 'real':
            # apply random rotation
            if self.rotate_about_z:
                R = torch.eye(3)
                angle = 2 * self.pi * torch.rand(1)
                c = torch.cos(angle)
                s = torch.sin(angle)

                # # z
                # R[0, 0] = c
                # R[0, 1] = -s
                # R[1, 0] = s
                # R[1, 1] = c

                # # x
                # R[1, 1] = c
                # R[1, 2] = -s
                # R[2, 1] = s
                # R[2, 2] = c

                # y
                R[0, 0] = c
                R[0, 2] = s
                R[2, 0] = -s
                R[2, 2] = c

            else:
                # R = transforms.random_rotation()
                R = torch.from_numpy(Rot.random().as_matrix()).to(dtype=torch.float32)

            model_mesh = model_mesh.rotate(R=R.numpy())

            pc_pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
            pc0 = torch.from_numpy(np.asarray(pc_pcd.points)).transpose(0, 1)  # (3, m)
            pc0 = pc0.to(torch.float)

            pc = self._get_depth_pc(rotated_model_mesh=model_mesh, num_points=self.num_of_points)
            del pc_pcd, model_mesh, R

        else:
            raise ValueError("ShapeNet: type not correctly specified.")

        return pc.T, pc0.T

    def _get_depth_pc(self, rotated_model_mesh, num_points):

        model_mesh = rotated_model_mesh

        # computing diameter
        diameter = np.linalg.norm(np.asarray(model_mesh.get_max_bound()) - np.asarray(model_mesh.get_min_bound()))

        num_sampled_points = 0
        factor_ = 5
        while num_sampled_points < num_points:

            # increasing the factor of sampled points
            factor_ = 2*factor_

            # sample a point cloud from the self.model_mesh
            pc_pcd_ = model_mesh.sample_points_uniformly(number_of_points=factor_*num_points)

            # take a depth image from a distance of the rotated self.model_mesh from self.camera_location
            beta = torch.rand(1, 1)
            camera_location_factor = beta * (self.radius_multiple[1] - self.radius_multiple[0]) + self.radius_multiple[0]
            camera_location_factor = camera_location_factor * diameter
            radius = get_radius(cam_location=camera_location_factor * self.camera_location.numpy(),
                                object_diameter=diameter)
            pc_pcd = get_depth_pcd(centered_pcd=pc_pcd_, camera=self.camera_location.numpy(), radius=radius)

            # converting pc to torch.tensor
            pc = torch.from_numpy(np.asarray(pc_pcd.points)).transpose(0, 1)  # (3, m)
            pc = pc.to(torch.float)
            num_sampled_points = pc.shape[-1]

        if num_sampled_points > num_points:
            idx_range = torch.arange(pc.shape[-1]).to(dtype=torch.float64)
            random_idx = torch.multinomial(idx_range, pc.shape[-1], replacement=False)
            pc = pc[:, random_idx]

        return pc

