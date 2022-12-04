import os
import numpy as np
import argparse
import open3d as o3d
from urllib.request import urlretrieve
import torch
import sys
sys.path.append("../..")

from c3po.baselines.fcgf.util.visualization import get_colored_point_cloud_feature
from c3po.baselines.fcgf.util.misc import extract_features
from c3po.baselines.fcgf.model.resunet import ResUNetBN2C
from c3po.datasets.shapenet import SE3PointCloudAll, DepthPCAll

def demo(config):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  checkpoint = torch.load(config.model)
  model = ResUNetBN2C(1, 32, normalize_feature=True, conv1_kernel_size=5, D=3)
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()

  model = model.to(device)

  if config.type == 'real':
      ds = DepthPCAll(dataset_len=1)
  elif config.type == 'sim':
      ds = SE3PointCloudAll(dataset_len=1)

  pc0, pc1, _, _, R, t = ds[0]

  # breakpoint( )
  xyz0_down, feature0 = extract_features(
      model,
      xyz=pc0.T,
      voxel_size=config.voxel_size,
      device=device,
      skip_check=True)

  xyz1_down, feature1 = extract_features(
      model,
      xyz=pc1.T,
      voxel_size=config.voxel_size,
      device=device,
      skip_check=True)

  vis_pcd0 = o3d.geometry.PointCloud()
  vis_pcd0.points = o3d.utility.Vector3dVector(xyz0_down)
  vis_pcd0 = get_colored_point_cloud_feature(vis_pcd0,
                                             feature0.detach().cpu().numpy(),
                                             config.voxel_size)

  vis_pcd1 = o3d.geometry.PointCloud()
  vis_pcd1.points = o3d.utility.Vector3dVector(xyz1_down)
  vis_pcd1 = get_colored_point_cloud_feature(vis_pcd1,
                                             feature1.detach().cpu().numpy(),
                                             config.voxel_size)

  o3d.visualization.draw_geometries([vis_pcd0, vis_pcd1])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-m',
      '--model',
      default='ResUNetBN2C-16feat-3conv.pth',
      type=str,
      help='path to latest checkpoint (default: None)')
  parser.add_argument(
      '--voxel_size',
      default=0.025,
      type=float,
      help='voxel size to preprocess point cloud')
  parser.add_argument(
      '--type',
      default='sim',
      type=str,
      help='specify:sim or real')

  config = parser.parse_args()
  demo(config)
