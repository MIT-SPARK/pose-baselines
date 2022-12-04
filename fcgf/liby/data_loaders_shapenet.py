import numpy as np
# import logging
# import sys
import yaml
import torch
import open3d as o3d
import MinkowskiEngine as ME
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

from c3po.datasets.shapenet import SE3PointCloudAll, DepthPCAll
from c3po.baselines.fcgf.liby.data_loaders import collate_pair_fn
from c3po.baselines.fcgf.util.pointcloud import get_matching_indices, make_open3d_point_cloud

expt_shapenet_dir = Path(__file__).parent.parent.parent.parent / 'expt_shapenet'


class ShapeNetDataset(torch.utils.data.Dataset):
  def __init__(self,
               type='sim',
               voxel_size=0.025,    # same as in config.py
               positive_pair_search_voxel_size_multiplier=1.5,   # same as in config.py
               dataset_length=2048,    # tunable parameter
               manual_seed=False
               ):

    self.type = type
    self.dataset_length = dataset_length
    self.voxel_size = voxel_size
    self.matching_search_voxel_size = voxel_size * positive_pair_search_voxel_size_multiplier

    if self.type == 'real':
        self.ds = DepthPCAll(dataset_len=dataset_length)
    elif self.type == 'sim':
        self.ds = SE3PointCloudAll(dataset_len=dataset_length)

    self.randg = np.random.RandomState()
    if manual_seed:
        self.reset_seed()

  def reset_seed(self, seed=0):
    # logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def __len__(self):

      return self.dataset_length

  def __getitem__(self, idx):

    # breakpoint()
    pc0, pc1, _, _, R, t = self.ds[idx]

    xyz0 = pc0.T.numpy()
    color0 = 0.5 * torch.ones_like(pc0.T).numpy()
    color0 += 0.001 * np.random.randn(*color0.shape)
    xyz1 = pc1.T.numpy()
    color1 = 0.5 * torch.ones_like(pc1.T).numpy()
    color1 += 0.001 * np.random.randn(*color1.shape)

    xyz0 = np.ascontiguousarray(xyz0)
    xyz1 = np.ascontiguousarray(xyz1)
    color0 = np.ascontiguousarray(color0)
    color1 = np.ascontiguousarray(color1)

    # breakpoint()
    trans = torch.eye(4)
    trans[:3, :3] = R
    trans[:3, 3:] = t
    trans = trans.numpy()
    trans = trans.astype('float64')

    matching_search_voxel_size = self.matching_search_voxel_size

    # Voxelization
    _, sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
    _, sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)

    # Make point clouds using voxelized points
    pcd0 = make_open3d_point_cloud(xyz0)
    pcd1 = make_open3d_point_cloud(xyz1)

    # Select features and points using the returned voxelized indices
    pcd0.colors = o3d.utility.Vector3dVector(color0[sel0])
    pcd1.colors = o3d.utility.Vector3dVector(color1[sel1])
    pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
    pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])
    # Get matches
    matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)

    # Get features
    npts0 = len(pcd0.colors)
    npts1 = len(pcd1.colors)

    feats_train0, feats_train1 = [], []

    feats_train0.append(np.ones((npts0, 1)))
    feats_train1.append(np.ones((npts1, 1)))

    feats0 = np.hstack(feats_train0)
    feats1 = np.hstack(feats_train1)

    # Get coords
    xyz0 = np.array(pcd0.points)
    xyz1 = np.array(pcd1.points)

    coords0 = np.floor(xyz0 / self.voxel_size)
    coords1 = np.floor(xyz1 / self.voxel_size)

    xyz0 = np.ascontiguousarray(xyz0)
    xyz1 = np.ascontiguousarray(xyz1)
    coords0 = np.ascontiguousarray(coords0)
    coords1 = np.ascontiguousarray(coords1)
    feats0 = np.ascontiguousarray(feats0)
    feats1 = np.ascontiguousarray(feats1)

    return (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans)


def make_data_loader(type, dataset_length, batch_size,
                     voxel_size, positive_pair_search_voxel_size_multiplier):

    ds = ShapeNetDataset(type=type,
                         voxel_size=voxel_size,
                         dataset_length=dataset_length,
                         positive_pair_search_voxel_size_multiplier=positive_pair_search_voxel_size_multiplier
                         )
    # breakpoint()
    # from c3po.utils.visualization_utils import display_two_pcs
    # (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans) = ds[0]
    # display_two_pcs(pc1=torch.from_numpy(xyz0).T.unsqueeze(0), pc2=torch.from_numpy(xyz1).T.unsqueeze(0))

    dl = torch.utils.data.DataLoader(ds,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     collate_fn=collate_pair_fn
                                     )

    return dl
