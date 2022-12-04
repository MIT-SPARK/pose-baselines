import numpy as np
import open3d as o3d
import torch

from utils_common import pos_tensor_to_o3d


# ransac from open3d
def ransac(source_points, target_points):
    """
    inputs:
    source_points   : torch.tensor of shape (3, m)
    target_points   : torch.tensor of shape (3, m)

    outputs:
    R   : torch.tensor of shape (3, 3)
    t   : torch.tensor of shape (3, 1)

    Note:
        Input and output will be on the same device, while compute will happen on cpu.

    """
    _, m = source_points.shape
    device_ = source_points.device

    # converting to open3d
    src = pos_tensor_to_o3d(pos=source_points.to('cpu'), estimate_normals=False)
    tar = pos_tensor_to_o3d(pos=target_points.to('cpu'), estimate_normals=False)

    # Initializing the correspondences
    a = torch.arange(0, m, 1).unsqueeze(0)
    c = torch.cat([a, a], dim=0).T
    d = c.numpy().astype('int32')
    corres_init = o3d.utility.Vector2iVector(d)

    # ransac from open3d
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source=src,
        target=tar,
        corres=corres_init,
        max_correspondence_distance=0.001)

    # extracting result
    T = result_ransac.transformation
    R_ = np.array(T[:3, :3])
    t_ = np.array(T[:3, 3])
    R = torch.from_numpy(R_)
    t = torch.from_numpy(t_)
    t = t.unsqueeze(-1)

    return R.to(device=device_), t.to(device=device_)


class RANSAC():
    """
    This code implements batch RANSAC for input, output given as torch.tensors.

    """

    def __init__(self, source_points):
        super().__init__()
        """
        source_points   : torch.tensor of shape (1, 3, m)

        """

        self.source_points = source_points.squeeze(0)
        self.device_ = source_points.device

    def forward(self, target_points):
        """
        input:
        target_points   : torch.tensor of shape (B, 3, n)

        output:
        R   : torch.tensor of shape (B, 3, 3)
        t   : torch.tensor of shape (B, 3, 1)

        """
        batch_size = target_points.shape[0]

        R = torch.zeros(batch_size, 3, 3).to(device=self.device_)
        t = torch.zeros(batch_size, 3, 1).to(device=self.device_)

        for b in range(batch_size):
            tar = target_points[b, ...]
            R_batch, t_batch = ransac(source_points=self.source_points, target_points=tar)
            R[b, ...] = R_batch
            t[b, ...] = t_batch

        return R, t

