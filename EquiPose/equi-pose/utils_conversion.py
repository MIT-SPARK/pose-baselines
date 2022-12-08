
import torch

import sys
sys.path.append("../../")

import shapenet
import ycb

from utils.extensions.chamfer_dist import ChamferFunction


class ChamferDistanceSqrt(torch.nn.Module):
    def __init__(self, ignore_zeros=False):
        super(ChamferDistanceSqrt, self).__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2, return_raw=False):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)
        if return_raw:
            return dist1, dist2
        else:
            return torch.mean(dist1) + torch.mean(dist2)


def removeTranslationShapenet(data):

    points = data['points'].squeeze(0).T
    xyz = data['xyz'].squeeze(0).T
    device_ = points.device
    t_gt = data['T'].squeeze(0).T

    # breakpoint()
    data['xyz'] = (xyz - t_gt).to(device=device_).T.unsqueeze(0)
    data['T'] = torch.zeros(1, 1, 3).to(device=device_)

    return data


def correctEquiPoseGT(data):
    """
    The modelnet40_complete/partial does not give the correct translation. We fix it with this function.

    """

    points = data['points'].squeeze(0).T
    xyz = data['xyz'].squeeze(0).T
    R_gt = data['R_gt'].squeeze(0)
    device_ = points.device

    points_c = points.mean(-1).unsqueeze(-1)
    xyz_c = xyz.mean(-1).unsqueeze(-1)

    # t_gt = xyz_c - R_gt @ points_c
    # data['T'] = t_gt.T.unsqueeze(0).to(device=device_)
    data['T'] = torch.zeros(1, 1, 3).to(device=device_)

    data['points'] = (points - points_c).T.unsqueeze(0).to(device=device_)
    data['xyz'] = (xyz - xyz_c).T.unsqueeze(0).to(device=device_)

    return data


class equiposeDataToStandardForm():
    def __init__(self):
        self.none = 0.0

    def __call__(self, data):
        data = correctEquiPoseGT(data)
        points = data['points'].squeeze(0).T
        xyz = data['xyz'].squeeze(0).T
        R_gt = data['R_gt'].squeeze(0)
        t_gt = data['T'].squeeze(0).T
        T_gt = torch.eye(4)
        T_gt[:3, :3] = R_gt
        T_gt[:3, 3:] = t_gt

        return points, xyz, T_gt


class ShapeNet(torch.utils.data.Dataset):
    def __init__(self, type, object, split, num_points, adv_option):

        if split == 'train':
            len_ = 2048
        elif split == 'test' or 'val':
            len_ = 512
        else:
            raise ValueError("ShapeNet: split not correctly specified.")

        self.ds = shapenet.ShapeNet(type=type,
                                    object=object,
                                    length=len_,
                                    num_points=num_points,
                                    adv_option=adv_option)

        self.class_name = object

    def __len__(self):

        return len(self.ds)

    def __getitem__(self, item):

        pc0, pc1, _, _, R, t = self.ds[item]

        # pcx = R.T @ (pc1 - t)   # equipose: we give them partial point cloud in canonical pose
        data_dict = {}
        data_dict['xyz'] = pc1.T.to(dtype=torch.float32)  # nx3 torch.tensor dtype=torch.float32
        data_dict['points'] = pc0.T.to(dtype=torch.float32)  # nx3 torch.tensor of dtype=torch.float32
        data_dict['label'] = torch.tensor([0.0])
        data_dict['R_gt'] = R
        data_dict['R'] = R      #TODO: need to address this. Check modelnet40_complete.py
        data_dict['T'] = t.T
        data_dict['fn'] = str(self.class_name)
        data_dict['id'] = str(self.class_name)
        data_dict['idx'] = str(self.class_name)
        data_dict['class'] = str(self.class_name)

        return data_dict


class YCB(torch.utils.data.Dataset):
    def __init__(self, type, object, split, num_points):

        if split == 'train':
            len_ = 2048
        elif split == 'test' or 'val':
            len_ = 512
        else:
            raise ValueError("YCB: split not correctly specified.")

        self.ds = ycb.YCB(type=type,
                          object=object,
                          length=len_,
                          num_points=num_points,
                          split=split)

        self.object = object

    def __len__(self):

        return len(self.ds)

    def __getitem__(self, item):

        pc0, pc1, _, _, R, t = self.ds[item]

        data_dict = {}
        data_dict['xyz'] = pc1.T.to(dtype=torch.float32)  # nx3 torch.tensor dtype=torch.float32
        data_dict['points'] = pc0.T.to(dtype=torch.float32)  # nx3 torch.tensor of dtype=torch.float32
        data_dict['label'] = torch.tensor([0.0])
        data_dict['R_gt'] = R
        data_dict['R'] = R  # TODO: need to address this. Check modelnet40_complete.py
        data_dict['T'] = t
        data_dict['fn'] = str(self.object)
        data_dict['id'] = str(self.object)
        data_dict['idx'] = str(self.object)
        data_dict['class'] = str(self.object)

        return data_dict

