import open3d as o3d
import os
# import teaserpp_python
import numpy as np
import torch
import copy
from urllib.request import urlretrieve
import sys
from pathlib import Path

sys.path.append("../..")
from c3po.baselines.fcgf.util.misc import extract_features
from c3po.baselines.fcgf.liby.eval import find_nn_gpu
import c3po.baselines.fcgf.util.transform_estimation as te
from c3po.baselines.fcgf.model.resunet import ResUNetBN2C
from c3po.baselines.teaser_utils.helpers import find_correspondences, get_teaser_solver, Rt2T
from c3po.utils.general import pos_tensor_to_o3d

FCGF_HOME = Path(__file__).parent / 'fcgf'


def find_corr(xyz0, xyz1, F0, F1, subsample_size=10000):
    subsample = len(F0) > subsample_size
    if subsample_size > 0 and subsample:
        N0 = min(len(F0), subsample_size)
        N1 = min(len(F1), subsample_size)
        inds0 = np.random.choice(len(F0), N0, replace=False)
        inds1 = np.random.choice(len(F1), N1, replace=False)
        F0, F1 = F0[inds0], F1[inds1]

    # Compute the nn
    nn_inds = find_nn_gpu(F0, F1, nn_max_n=500) # self.config.nn_max_n = 500
    if subsample_size > 0 and subsample:
        return xyz0[inds0], xyz1[inds1[nn_inds]]
    else:
        return xyz0, xyz1[nn_inds]


def teaser_fcgf_icp(source_points, target_points, voxel_size=0.025, model=None,
                    pre_trained_3dmatch=False, visualize=False, icp=True):
  """
  source_points : torch.tensor of shape (3, n)
  target_points : torch.tensor of shape (3, m)

  source_down   : torch.tensor of shape (3, l)
  target_down   : torch.tensor of shape (3, k)
  source_feat   : torch.tensor of shape (d, l)
  target_feat   : torch.tensor of shape (d, k)
  """
  # # setup device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # visualize
  if visualize:
      src_ = pos_tensor_to_o3d(source_points, estimate_normals=False)
      tar_ = pos_tensor_to_o3d(target_points, estimate_normals=False)
      src_.paint_uniform_color([0.0, 0.0, 1.0])  # show src_ in blue
      tar_.paint_uniform_color([1.0, 0.0, 0.0])  # show tar_ in red
      o3d.visualization.draw_geometries([src_, tar_])  # plot src_ and tar_

  # load pre-trained model
  if model is None:
      if not os.path.isfile(FCGF_HOME / 'ResUNetBN2C-16feat-3conv.pth'):
          print('Downloading weights...')
          urlretrieve(
              "https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth",
              FCGF_HOME / 'ResUNetBN2C-16feat-3conv.pth')

      checkpoint = torch.load(FCGF_HOME / 'ResUNetBN2C-16feat-3conv.pth')
      model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)
      model.load_state_dict(checkpoint['state_dict'])
      model.eval()
      model = model.to(device)

  elif pre_trained_3dmatch:
      checkpoint = torch.load(model)
      model = ResUNetBN2C(1, 16, normalize_feature=True, conv1_kernel_size=3, D=3)
      model.load_state_dict(checkpoint['state_dict'])
      model.eval()
      model = model.to(device)

  else:
      checkpoint = torch.load(model)
      model = ResUNetBN2C(1, 32, normalize_feature=True, conv1_kernel_size=5, D=3)
      model.load_state_dict(checkpoint['state_dict'])
      model.eval()
      model = model.to(device)


  # extracting features
  # src = source_points.transpose(-1, -2).to('cpu').numpy()
  # tar = target_points.transpose(-1, -2).to('cpu').numpy()
  src = source_points.transpose(-1, -2)
  tar = target_points.transpose(-1, -2)

  src_down, src_feature = extract_features(
      model,
      xyz=src,
      voxel_size=voxel_size,
      device=device,
      skip_check=True)

  tar_down, tar_feature = extract_features(
      model,
      xyz=tar,
      voxel_size=voxel_size,
      device=device,
      skip_check=True)
  src_down = src_down.T
  tar_down = tar_down.T

  # visualize
  if visualize:
      src_ = pos_tensor_to_o3d(src_down, estimate_normals=False)
      tar_ = pos_tensor_to_o3d(tar_down, estimate_normals=False)
      src_.paint_uniform_color([0.0, 0.0, 1.0])  # show src_ in blue
      tar_.paint_uniform_color([1.0, 0.0, 0.0])  # show tar_ in red
      o3d.visualization.draw_geometries([src_, tar_])  # plot src_ and tar_

  # establish correspondences by nearest neighbour search in feature space
  # breakpoint()
  src_corr, tar_corr = find_corr(xyz0=src_down.T,
                                 xyz1=tar_down.T,
                                 F0=src_feature,
                                 F1=tar_feature)
  # src_corr = torch.from_numpy(src_corr).to(device=device)
  # tar_corr = torch.from_numpy(tar_corr).to(device=device)
  # corrs_src, corrs_tar = find_correspondences(src_feature.to('cpu').detach().numpy(),
  #                                             tar_feature.to('cpu').detach().numpy(),
  #                                             mutual_filter=True)
  #
  # src_corr = src_down[:, corrs_src]  # np array of size 3 by num_corrs
  # tar_corr = tar_down[:, corrs_tar]  # np array of size 3 by num_corrs
  src_corr = src_corr.T
  tar_corr = tar_corr.T

  # ToDo: This only makes sure that teaser doesn't compute for infinite time.
  # if src_corr.shape[1] > 3000:
  #     src_corr = src_corr[:, 3000]
  #     tar_corr = tar_corr[:, 3000]

  if visualize:
      num_corrs = src_corr.shape[1]
      print(f'FCGH generates {num_corrs} putative correspondences.')

  # visualize the point clouds together with feature correspondences
  if visualize:
      points = np.concatenate((src_corr.T.to('cpu').detach().numpy(), tar_corr.T.to('cpu').detach().numpy()), axis=0)
      lines = []
      for i in range(num_corrs):
          lines.append([i, i + num_corrs])
      colors = [[0, 1, 0] for i in range(len(lines))]  # lines are shown in green
      line_set = o3d.geometry.LineSet(
          points=o3d.utility.Vector3dVector(points),
          lines=o3d.utility.Vector2iVector(lines),
      )
      line_set.colors = o3d.utility.Vector3dVector(colors)
      o3d.visualization.draw_geometries([src_, tar_, line_set])

  # robust global registration using TEASER++
  noise_bound = voxel_size
  teaser_solver = get_teaser_solver(noise_bound)
  teaser_solver.solve(src_corr.to('cpu').detach().numpy(), tar_corr.to('cpu').detach().numpy())
  solution = teaser_solver.getSolution()
  R_teaser = solution.rotation
  t_teaser = solution.translation
  T_teaser = Rt2T(R_teaser, t_teaser)

  # local refinement using ICP
  if icp:
      src_down_ = pos_tensor_to_o3d(src_down.to('cpu').detach(), estimate_normals=False)
      tar_down_ = pos_tensor_to_o3d(tar_down.to('cpu').detach(), estimate_normals=False)
      icp_sol = o3d.pipelines.registration.registration_icp(
          src_down_, tar_down_, noise_bound, T_teaser,
          o3d.pipelines.registration.TransformationEstimationPointToPoint(),
          o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
      T_icp = icp_sol.transformation

  else:
    T_icp = T_teaser

  # visualize the registration after ICP refinement
  if visualize:
      src_down_ = pos_tensor_to_o3d(src_down.to('cpu').detach(), estimate_normals=True)
      tar_down_ = pos_tensor_to_o3d(tar_down.to('cpu').detach(), estimate_normals=True)
      src_down_T_icp = copy.deepcopy(src_down_).transform(T_icp)
      src_down_T_icp.paint_uniform_color([0.0, 0.0, 1.0])
      tar_down_.paint_uniform_color([1.0, 0.0, 0.0])
      o3d.visualization.draw_geometries([src_down_T_icp, tar_down_])

  T_ = copy.deepcopy(T_icp)
  T = torch.from_numpy(T_)

  return T[:3, :3], T[:3, 3:4]


class TEASER_FCGF_ICP():
    """
    This code implements batch TEASER++ (with correspondences using FCGF) + ICP
    """

    def __init__(self, source_points, model=None, pre_trained_3dmatch=False, voxel_size=0.025, visualize=False):
        """
        source_points   : torch.tensor of shape (1, 3, m)

        """

        self.source_points = source_points
        self.viz = visualize
        self.model = model
        self.voxel_size = voxel_size
        self.device_ = source_points.device
        self.pre_trained_3dmatch = pre_trained_3dmatch

    def forward(self, target_points):
        """
        input:
        target_points   : torch.tensor of shape (B, 3, n)

        output:
        R   : torch.tensor of shape (B, 3, 3)
        t   : torch.tensor of shape (B, 3, 1)

        """
        batch_size = target_points.shape[0]
        src = self.source_points.squeeze(0)
        viz = self.viz
        vox_size = self.voxel_size
        model = self.model
        pre_trained_3dmatch = self.pre_trained_3dmatch

        R = torch.zeros(batch_size, 3, 3).to(device=self.device_)
        t = torch.zeros(batch_size, 3, 1).to(device=self.device_)

        for b in range(batch_size):
            tar = target_points[b, ...]

            # pruning the tar of all zero points
            idx = torch.any(tar == 0, 0)
            tar_new = tar[:, torch.logical_not(idx)]

            # teaser + fpfh + icp
            R_batch, t_batch = teaser_fcgf_icp(source_points=src,
                                               target_points=tar_new,
                                               voxel_size=vox_size,
                                               model=model,
                                               pre_trained_3dmatch=pre_trained_3dmatch,
                                               visualize=viz)

            R[b, ...] = R_batch
            t[b, ...] = t_batch

        return R, t


if __name__ == "__main__":
    # Load and visualize two point clouds from 3DMatch dataset
    src = o3d.io.read_point_cloud('../fpfh_teaser/test_data/cloud_bin_0.ply')
    tar = o3d.io.read_point_cloud('../fpfh_teaser/test_data/cloud_bin_4.ply')

    # Convert o3d PointCloud to torch.tensors
    source_points = torch.from_numpy(np.asarray(src.points).T)
    target_points = torch.from_numpy(np.asarray(tar.points).T)

    # Calling teaser (with FPFH correspondences) + ICP
    R, t = teaser_fcgf_icp(source_points, target_points, voxel_size=0.025, visualize=True)


