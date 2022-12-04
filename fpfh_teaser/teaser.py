import torch
import teaserpp_python


def teaser(source_points, target_points):
    """
    inputs:
    source_points   : torch.tensor of shape (3, m)
    target_points   : torch.tensor of shape (3, n)

    outputs:
    R   : torch.tensor of shape (3, 3)
    t   : torch.tensor of shape (3, 1)

    Note:
        Input and output will be on the same device, while compute will happen on cpu.

    """
    device_ = source_points.device
    # print("Here!")

    # convert source_points, target_points to numpy src, tar
    src = source_points.squeeze(0).to('cpu').numpy()
    tar = target_points.to('cpu').numpy()

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.05
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(src, tar)

    solution = solver.getSolution()

    R = solution.rotation
    t = solution.translation
    R = torch.from_numpy(R)
    t = torch.from_numpy(t)
    t = t.unsqueeze(-1)

    return R.to(device=device_), t.to(device=device_)


class TEASER():
    """
    This code implements batch TEASER++ for input, output given as torch.tensors.
    """
    def __init__(self, source_points):
        super().__init__()
        """
        source_points   : torch.tensor of shape (1, 3, m)

        """

        self.source_points = source_points
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

            # pruning the tar of all zero points
            idx = torch.any(tar == 0, 0)
            tar_new = tar[:, torch.logical_not(idx)]

            # teaser
            R_batch, t_batch = teaser(source_points=self.source_points, target_points=tar_new)
            R[b, ...] = R_batch
            t[b, ...] = t_batch

        return R, t

