
import torch
import open3d as o3d
import numpy as np

from utils_common import pos_tensor_to_o3d
from teaser import TEASER
from ransac import RANSAC


def icp(source_points, target_points, R0, t0):
    """
    inputs:
    source_points   : torch.tensor of shape (3, m)
    target_points   : torch.tensor of shape (3, n)
    R0              : torch.tensor of shape (3, 3)
    t0              : torch.tensor of shape (3, 1)

    outputs:
    R   : torch.tensor of shape (3, 3)
    t   : torch.tensor of shape (3, 1)

    Note:
        Input and output will be on the same device, while compute will happen on cpu.

    """

    # converting to open3d
    src = pos_tensor_to_o3d(pos=source_points.to('cpu'), estimate_normals=False)
    tar = pos_tensor_to_o3d(pos=target_points.to('cpu'), estimate_normals=False)

    # transformation
    T = torch.zeros(4, 4).to('cpu')
    T[:3, :3] = R0.to('cpu')
    T[:3, 3:] = t0.to('cpu')
    T = T.numpy()

    # icp from open3d
    reg_p2p = o3d.pipelines.registration.registration_icp(src, tar, 0.01, T,
                                                          o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                          o3d.pipelines.registration.ICPConvergenceCriteria(
                                                              max_iteration=200))

    # extracting results
    T = reg_p2p.transformation
    R_ = np.array(T[:3, :3])
    t_ = np.array(T[:3, 3])
    R = torch.from_numpy(R_)
    t = torch.from_numpy(t_)
    t = t.unsqueeze(-1)

    return R, t


class ICP():
    """
    This code implements batch ICP for input, output given as torch.tensors.
    """
    def __init__(self, source_points):
        super().__init__()
        """
        source_points   : torch.tensor of shape (1, 3, m)

        """

        self.source_points = source_points.squeeze(0)

    def forward(self, target_points, R0, t0):
        """
        input:
        target_points   : torch.tensor of shape (B, 3, n)
        R0              : torch.tensor of shape (B, 3, 3)
        t0              : torch.tensor of shape (B, 3, 1)

        output:
        R   : torch.tensor of shape (B, 3, 3)
        t   : torch.tensor of shape (B, 3, 1)

        """
        batch_size = target_points.shape[0]

        R = torch.zeros_like(R0)
        t = torch.zeros_like(t0)

        for b in range(batch_size):

            # removes the padded zero points
            tarX = target_points[b, ...]
            idx = torch.sum(tarX == 0, dim=0) == 3
            tar = tarX[:, torch.logical_not(idx)]  # (3, n')

            # icp
            R_batch, t_batch = icp(source_points=self.source_points,
                                   target_points=tar,
                                   R0=R0[b, ...],
                                   t0=t0[b, ...])
            R[b, ...] = R_batch
            t[b, ...] = t_batch

        return R, t


class RANSACwICP():
    def __init__(self, cad_models, model_keypoints):
        super().__init__()
        """
        cad_models : torch.tensor of shape (1, 3, m)
        model_keypoints     : torch.tensor of shape (1, 3, K)

        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints

        self.RANSAC = RANSAC(source_points=self.model_keypoints)
        self.ICP = ICP(source_points=self.cad_models)

    def forward(self, input_point_cloud, detected_keypoints):
        """
        input_point_cloud   : torch.tensor of shape (B, 3, n)
        detected_keypoints  : torch.tensor of shape (B, 3, K)

        output:
        predicted_point_cloud   : torch.tensor of shape (B, 3, m)
        rotation                : torch.tensor of shape (B, 3, 3)
        translation             : torch.tensor of shape (B, 3, 1)

        """

        _, _, m = input_point_cloud.shape

        # centering. This considers that we may have padded zero points.
        num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
        num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
        num_nonzero_pts = m - num_zero_pts
        num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)

        center = torch.sum(input_point_cloud, dim=-1) / num_nonzero_pts
        center = center.unsqueeze(-1)
        pc_centered = input_point_cloud - center
        kp_centered = detected_keypoints - center

        # global registration
        R0, t0 = self.RANSAC.forward(target_points=kp_centered)

        # icp
        R, t = self.ICP.forward(target_points=pc_centered, R0=R0, t0=t0)

        # re-centering
        t = t + center

        return R @ self.cad_models + t, R, t


class TEASERwICP():
    def __init__(self, cad_models, model_keypoints):
        super().__init__()
        """
        cad_models : torch.tensor of shape (1, 3, m)
        model_keypoints     : torch.tensor of shape (1, 3, K)

        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints

        self.TEASER = TEASER(source_points=self.model_keypoints)
        self.ICP = ICP(source_points=self.cad_models)

    def forward(self, input_point_cloud, detected_keypoints):
        """
        input_point_cloud   : torch.tensor of shape (B, 3, n)
        detected_keypoints  : torch.tensor of shape (B, 3, K)

        output:
        predicted_point_cloud   : torch.tensor of shape (B, 3, m)
        rotation                : torch.tensor of shape (B, 3, 3)
        translation             : torch.tensor of shape (B, 3, 1)

        """

        _, _, m = input_point_cloud.shape

        # centering. This considers that we may have padded zero points.
        num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
        num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
        num_nonzero_pts = m - num_zero_pts
        num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)

        center = torch.sum(input_point_cloud, dim=-1) / num_nonzero_pts
        center = center.unsqueeze(-1)
        pc_centered = input_point_cloud - center
        kp_centered = detected_keypoints - center

        # global registration
        R0, t0 = self.TEASER.forward(target_points=kp_centered)

        # icp
        R, t = self.ICP.forward(target_points=pc_centered, R0=R0, t0=t0)

        # re-centering
        t = t + center

        return R @ self.cad_models + t, R, t


class wICP():
    def __init__(self, cad_models, model_keypoints):
        super().__init__()
        """
        cad_models : torch.tensor of shape (1, 3, m)
        model_keypoints     : torch.tensor of shape (1, 3, K)

        """
        self.cad_models = cad_models
        self.model_keypoints = model_keypoints

        self.TEASER = TEASER(source_points=self.model_keypoints)
        self.ICP = ICP(source_points=self.cad_models)

    def forward(self, input_point_cloud, R0, t0):
        """
        input_point_cloud   : torch.tensor of shape (B, 3, n)
        R0                  : torch.tensor of shape (B, 3, 3)
        t0                  : torch.tensor of shape (B, 3, 1)

        output:
        predicted_point_cloud   : torch.tensor of shape (B, 3, m)
        rotation                : torch.tensor of shape (B, 3, 3)
        translation             : torch.tensor of shape (B, 3, 1)

        """
        _, _, m = input_point_cloud.shape

        # centering. This considers that we may have padded zero points.
        num_zero_pts = torch.sum(input_point_cloud == 0, dim=1)
        num_zero_pts = torch.sum(num_zero_pts == 3, dim=1)
        num_nonzero_pts = m - num_zero_pts
        num_nonzero_pts = num_nonzero_pts.unsqueeze(-1)

        center = torch.sum(input_point_cloud, dim=-1) / num_nonzero_pts
        center = center.unsqueeze(-1)
        pc_centered = input_point_cloud - center
        t0 = t0 - center

        # icp
        R, t = self.ICP.forward(target_points=pc_centered, R0=R0, t0=t0)

        # re-centering
        t = t + center

        return R @ self.cad_models + t, R, t