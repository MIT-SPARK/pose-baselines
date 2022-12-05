
import numpy as np
import yaml
import json
import torch
import open3d as o3d
from pathlib import Path
import copy
from scipy.spatial.transform import Rotation as Rot
# from pytorch3d import transforms, ops
import random
BASE_DIR = Path(__file__).parent.parent.parent

# expt_shapenet_dir = Path(__file__).parent.parent.parent.parent / 'expt_shapenet'
ANNOTATIONS_FOLDER: str = str(BASE_DIR) + '/data_shapenet/KeypointNet/KeypointNet/annotations/'
PCD_FOLDER_NAME: str = str(BASE_DIR) + '/data_shapenet/KeypointNet/KeypointNet/pcds/'
MESH_FOLDER_NAME: str = str(BASE_DIR) + '/data_shapenet/KeypointNet/ShapeNetCore.v2.ply/'
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


class SE3PointCloud(torch.utils.data.Dataset):
    """
    Randomly generates ShapeNet object point clouds and their SE3 transformations.

    Returns a batch of
        pc1, pc2, kp1, kp2, rotation, translation

    """
    def __init__(self, class_name, num_of_points=1024, dataset_len=2048,
                 dir_location='../../data/learning-objects/keypointnet_datasets/'):
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
        R = torch.from_numpy(Rot.random().as_matrix()).to(dtype=torch.float32)
        # t = torch.rand(3, 1)
        t = torch.zeros(3, 1)

        pc1_pcd = model_mesh.sample_points_uniformly(number_of_points=self.num_of_points)
        pc1 = torch.from_numpy(np.asarray(pc1_pcd.points)).transpose(0, 1)
        pc1 = pc1.to(torch.float)

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
                 num_of_points2=2048, dataset_len=10000, rotate_about_z=False):
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
        # t = torch.rand(3, 1)
        t = torch.zeros(3, 1)
        pc2 = pc2 + t
        kp2 = R @ kp1 + t

        pc1 = R.T @ (pc2 - t)

        return (pc1, pc2, kp1, kp2, R, t)
        # return data_dict


class ShapeNetObjectDataset(torch.utils.data.Dataset):
  def __init__(self,
               split,
               type='depth',
               object='airplane'
               ):

    self.type = type
    self.class_name = object
    self.class_id = CLASS_ID[self.class_name]
    self.model_id = CLASS_MODEL_ID[self.class_name]
    self.n_points = 1024

    if split == 'train':
        self.dataset_length = 2048
    elif split == 'test':
        self.dataset_length = 512
        np.random.seed(seed=0)

    elif split == 'val':
        self.dataset_length = 512

    if self.type == 'depth':
        self.ds = DepthPC(class_name=self.class_name,
                          dataset_len=self.dataset_length,
                          num_of_points1=self.n_points,
                          num_of_points2=10*self.n_points)
    elif self.type == 'full':
        self.ds = SE3PointCloud(class_name=self.class_name,
                                dataset_len=self.dataset_length,
                                num_of_points=self.n_points)

  def __len__(self):

      return self.dataset_length

  def __getitem__(self, idx):

    if self.type == 'full':
        pc0, pc1, _, _, R, t = self.ds[idx]
    elif self.type == 'depth':
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

    # xyz0 = pc0.T.numpy()
    # xyz1 = pc1.T.numpy()
    #
    # xyz0 = np.ascontiguousarray(xyz0)
    # xyz1 = np.ascontiguousarray(xyz1)
    #
    # # breakpoint()
    # trans = torch.eye(4)
    # trans[:3, :3] = R
    # trans[:3, 3:] = t
    # trans = trans.numpy()

    data_dict = {}
    data_dict['xyz'] = pc1.T.to(dtype=torch.float32)  # nx3 torch.tensor dtype=torch.float32
    data_dict['points'] = pc0.T.to(dtype=torch.float32)  # nx3 torch.tensor of dtype=torch.float32
    data_dict['label'] = torch.tensor([0.0])
    data_dict['R_gt'] = R
    data_dict['R'] = R
    data_dict['T'] = t
    data_dict['fn'] = str(self.class_name)
    data_dict['id'] = str(self.model_id)
    data_dict['idx'] = str(self.model_id)
    data_dict['class'] = str(self.class_name)

    return data_dict


