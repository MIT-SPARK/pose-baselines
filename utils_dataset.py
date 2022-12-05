
import torch
import numpy as np


# credit: pointnetlk_revisited. This induces rotation errors, which we use to create "easy" dataset.
class RandomTransformSE3:
    """ randomly generate rigid transformations """

    def __init__(self, mag=0.8, mag_randomly=True):
        # mag=0.8 is the pointnetlk_revisited dataset
        self.mag = mag
        self.randomly = mag_randomly
        self.gt = None
        self.igt = None

    def generate_transform(self):
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp

        return x

    def apply_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6], twist params
        g = self.exp(x).to(p0)  # [1, 4, 4]
        gt = self.exp(-x).to(p0)  # [1, 4, 4]
        p1 = self.rot_transform(g, p0)
        self.gt = gt  # p1 --> p0
        self.igt = g  # p0 --> p1

        return p1

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)

    def rot_transform(self, g, a):
        # g : SE(3),  B x 4 x 4
        # a : R^3,    B x N x 3
        g_ = g.view(-1, 4, 4)
        R = g_[:, 0:3, 0:3].contiguous().view(*(g.size()[0:-2]), 3, 3)
        p = g_[:, 0:3, 3].contiguous().view(*(g.size()[0:-2]), 3)
        if len(g.size()) == len(a.size()):
            a = a.transpose(1, 2)
            b = R.matmul(a) + p.unsqueeze(-1)
        else:
            b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p
        return b

    # functions for exp map
    def exp(self, x):
        x_ = x.view(-1, 6)
        w, v = x_[:, 0:3], x_[:, 3:6]
        t = w.norm(p=2, dim=1).view(-1, 1, 1)  # norm of rotation
        W = self.mat_so3(w)
        S = W.bmm(W)
        I = torch.eye(3).to(w)

        # Rodrigues' rotation formula.
        R = I + self.sinc1(t) * W + self.sinc2(t) * S
        V = I + self.sinc2(t) * W + self.sinc3(t) * S

        p = V.bmm(v.contiguous().view(-1, 3, 1))

        z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(x_.size(0), 1, 1).to(x)
        Rp = torch.cat((R, p), dim=2)
        g = torch.cat((Rp, z), dim=1)

        return g.view(*(x.size()[0:-1]), 4, 4)

    def sinc1(self, t):
        """ sinc1: t -> sin(t)/t """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t[s] ** 2
        r[s] = 1 - t2 / 6 * (1 - t2 / 20 * (1 - t2 / 42))  # Taylor series O(t^8)
        r[c] = torch.sin(t[c]) / t[c]

        return r

    def sinc2(self, t):
        """ sinc2: t -> (1 - cos(t)) / (t**2) """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t ** 2
        r[s] = 1 / 2 * (1 - t2[s] / 12 * (1 - t2[s] / 30 * (1 - t2[s] / 56)))  # Taylor series O(t^8)
        r[c] = (1 - torch.cos(t[c])) / t2[c]

        return r

    def sinc3(self, t):
        """ sinc3: t -> (t - sin(t)) / (t**3) """
        e = 0.01
        r = torch.zeros_like(t)
        a = torch.abs(t)

        s = a < e
        c = (s == 0)
        t2 = t[s] ** 2
        r[s] = 1 / 6 * (1 - t2 / 20 * (1 - t2 / 42 * (1 - t2 / 72)))  # Taylor series O(t^8)
        r[c] = (t[c] - torch.sin(t[c])) / (t[c] ** 3)

        return r

    def mat_so3(self, x):
        # x: [*, 3]
        # X: [*, 3, 3]
        x_ = x.view(-1, 3)
        x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
        O = torch.zeros_like(x1)

        X = torch.stack((
            torch.stack((O, -x3, x2), dim=1),
            torch.stack((x3, O, -x1), dim=1),
            torch.stack((-x2, x1, O), dim=1)), dim=1)
        return X.view(*(x.size()[0:-1]), 3, 3)


# credit: created from pointnetlk_revisited code.
class PointRegistrationEasy(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform=RandomTransformSE3(), sigma=0.00, clip=0.00):
        self.dataset = dataset
        self.transf = rigid_transform
        self.sigma = sigma
        self.clip = clip

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pc1, pc2, kp1, kp2, R, t = self.dataset[index]

        pc2_ = R.T @ (pc2 - t)
        kp2_ = R.T @ (kp2 - t)

        # breakpoint()
        new_pc2 = self.transf(pc2_.T).T
        new_T = self.transf.igt.squeeze(0)
        new_R = new_T[:3, :3]
        new_t = new_T[:3, 3:]
        new_kp2 = new_R @ kp2_ + new_t

        # pm, p0, T = self.dataset[index]  # one point cloud
        # assumes: p0 is the full point cloud, pm is the depth point cloud of p0. no rotation change.
        # p_ = pm
        # p1 = self.transf(p_)
        # igt = self.transf.igt.squeeze(0)
        # p0 = pm
        # del p_, pm

        # p0: template, p1: source, igt:transform matrix from p0 to p1
        # return p0, p1, igt

        return (pc1, new_pc2, kp1, new_kp2, new_R, new_t)


# credit: created from deepgmr code.
class PointRegistrationMedium(torch.utils.data.Dataset):
    def __init__(self, ds, max_angle=180.0, max_dist=0.5):
        # max_angle = 180.0 : deepgmr default
        # max_dist = 0.5    : deepgmr default

        self.ds = ds
        self.max_angle = max_angle
        self.max_dist = max_dist

    def __len__(self):
        return self.ds.__len__()

    def __getitem__(self, item):

        pc1, pc2, kp1, kp2, R, t = self.ds[item]

        pc2_ = R.T @ (pc2 - t)
        kp2_ = R.T @ (kp2 - t)

        # breakpoint()
        new_T = self.random_pose()
        new_T = new_T.to(dtype=pc1.dtype)
        new_R = new_T[:3, :3]
        new_t = new_T[:3, 3:]

        new_pc2 = new_R @ pc2_ + new_t
        new_kp2 = new_R @ kp2_ + new_t

        return (pc1, new_pc2, kp1, new_kp2, new_R, new_t)

    def random_pose(self):
        R = self.random_rotation()
        t = self.random_translation()
        return torch.from_numpy(np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0))

    def random_rotation(self):
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        angle = np.random.rand() * self.max_angle
        A = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
        return torch.from_numpy(R)

    def random_translation(self):
        t = np.random.randn(3)
        t /= np.linalg.norm(t)
        t *= np.random.rand() * self.max_dist
        return torch.from_numpy(np.expand_dims(t, 1))