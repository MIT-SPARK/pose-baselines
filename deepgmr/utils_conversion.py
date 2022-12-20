
import torch
import numpy as np
import pickle
from tqdm import tqdm
import argparse

from torch.utils.data import Dataset

from data import get_rri, get_rri_cuda, jitter_pcd

import sys
sys.path.append("../")
import shapenet
import ycb

class deepgmrDataToStandardFormat:
    """
    Converts DeepGMR outputs to Standard Form

    """
    def __init__(self):
        self.none = 1.0

    def __call__(self, data):
        X, Y, T = data

        X_ = torch.tensor(X[:, :3].T)
        Y_ = torch.tensor(Y[:, :3].T)
        T_ = torch.tensor(T)

        return X_, Y_, T_


class ShapeNetDataset(torch.utils.data.Dataset):
    """
    Wraper for shapenet datasets in shapenet.py, for DeepGMR training.

    """
    def __init__(self, args, type, from_file=False, filename=None, adv_option='hard'):

        assert adv_option in ['hard', 'medium', 'easy']
            # hard: c3po rotation errors
            # easy: lk rotation errors
            # medium: deepgmr rotation errors

        self.type = type
        self.adv_option = adv_option
        self.use_rri = args.use_rri
        self.k = args.k if self.use_rri else None
        self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri
        self.filename = filename
        self.from_file = from_file

        if from_file and filename is not None:
            with open(self.filename, 'rb') as f:
                self.data_ = pickle.load(f)

        else:
            if args.object == 'all':
                object_ = 'airplane'
            else:
                object_ = args.object
            self.ds = shapenet.ShapeNet(type=type, object=object_, length=args.shapenet_ds_len,
                                        num_points=args.n_points, adv_option=adv_option)

    def __len__(self):
        if self.from_file:
            len_ = len(self.data_)
        else:
            len_ = len(self.ds)

        return len_

    def __getitem__(self, item):
        if self.from_file:
            pc0, pc1, _, _, R, t = self.data_[item]
        else:
            pc0, pc1, _, _, R, t = self.ds[item]

        xyz0 = pc0.T.numpy()
        xyz1 = pc1.T.numpy()

        xyz0 = np.ascontiguousarray(xyz0)
        xyz1 = np.ascontiguousarray(xyz1)

        # xyz0 = jitter_pcd(xyz0)
        # xyz1 = jitter_pcd(xyz1)

        # breakpoint()
        trans = torch.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:] = t
        trans = trans.numpy()

        if self.use_rri:
            xyz0 = np.concatenate([xyz0, self.get_rri(xyz0 - xyz0.mean(axis=0), self.k)], axis=1)
            xyz1 = np.concatenate([xyz1, self.get_rri(xyz1 - xyz1.mean(axis=0), self.k)], axis=1)
        # same points on both pc
        # breakpoint()

        return xyz0.astype('float32'), xyz1.astype('float32'), trans.astype('float32')

    def save_dataset(self, filename):
        from utils_common import display_two_pcs
        data_ = []
        len_ = len(self.ds)
        for i in tqdm(range(len_)):
            # breakpoint()
            data = self.ds[i]
            data_.append(data)

        with open(filename, 'wb') as f:
            pickle.dump(data_, f, protocol=pickle.HIGHEST_PROTOCOL)


class YCBDataset(torch.utils.data.Dataset):
    """
    Wraper for shapenet datasets in shapenet.py, for DeepGMR training.

    """
    def __init__(self, args, type, split, from_file=False, filename=None, adv_option='hard'):

        assert adv_option in ['hard', 'medium', 'easy']
            # hard: c3po rotation errors
            # easy: lk rotation errors
            # medium: deepgmr rotation errors
            # note: hard is default. in the dataset, it doesn't use any wrappers.
            # node: only use the hard option.

        self.type = type
        self.adv_option = adv_option
        self.use_rri = args.use_rri
        self.k = args.k if self.use_rri else None
        self.get_rri = get_rri_cuda if torch.cuda.is_available() else get_rri
        self.filename = filename
        self.from_file = from_file
        self.split = split

        if from_file and filename is not None:
            with open(self.filename, 'rb') as f:
                self.data_ = pickle.load(f)

        else:
            self.ds = ycb.YCB(type=type, object=args.object, length=args.shapenet_ds_len,
                              num_points=args.n_points, adv_option=adv_option, split=split)

    def __len__(self):
        if self.from_file:
            len_ = len(self.data_)
        else:
            len_ = len(self.ds)

        return len_

    def __getitem__(self, item):
        if self.from_file:
            pc0, pc1, _, _, R, t = self.data_[item]
        else:
            pc0, pc1, _, _, R, t = self.ds[item]

        xyz0 = pc0.T.numpy()
        xyz1 = pc1.T.numpy()

        xyz0 = np.ascontiguousarray(xyz0)
        xyz1 = np.ascontiguousarray(xyz1)

        # xyz0 = jitter_pcd(xyz0)
        # xyz1 = jitter_pcd(xyz1)
        # breakpoint()
        trans = torch.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:] = t
        trans = trans.numpy()

        if self.use_rri:
            xyz0 = np.concatenate([xyz0, self.get_rri(xyz0 - xyz0.mean(axis=0), self.k)], axis=1)
            xyz1 = np.concatenate([xyz1, self.get_rri(xyz1 - xyz1.mean(axis=0), self.k)], axis=1)
        # same points on both pc
        # breakpoint()

        return xyz0.astype('float32'), xyz1.astype('float32'), trans.astype('float32')

    def save_dataset(self, filename):
        from utils_common import display_two_pcs
        data_ = []
        len_ = len(self.ds)
        for i in tqdm(range(len_)):
            # breakpoint()
            data = self.ds[i]
            data_.append(data)

        with open(filename, 'wb') as f:
            pickle.dump(data_, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--use_rri', action='store_true')
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--object', type=str, default='all')  # shapenet object/class_name
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--shapenet_ds_len', type=int, default=2048)

    args = parser.parse_args()

    # shapenet.real
    # shapenet.real.hard
    ds = ShapeNetDataset(args, type='real', adv_option='hard')
    filename_ = 'data/train/shapenet.real.hard.pkl'
    ds.save_dataset(filename_)

    # shapenet.real.medium
    ds = ShapeNetDataset(args, type='real', adv_option='medium')
    filename_ = 'data/train/shapenet.real.medium.pkl'
    ds.save_dataset(filename_)

    # shapenet.real.easy
    ds = ShapeNetDataset(args, type='real', adv_option='easy')
    filename_ = 'data/train/shapenet.real.easy.pkl'
    ds.save_dataset(filename_)

    # shapenet.sim
    # shapenet.sim.hard
    ds = ShapeNetDataset(args, type='sim', adv_option='hard')
    filename_ = 'data/train/shapenet.sim.hard.pkl'
    ds.save_dataset(filename_)

    # shapenet.sim.medium
    ds = ShapeNetDataset(args, type='sim', adv_option='medium')
    filename_ = 'data/train/shapenet.sim.medium.pkl'
    ds.save_dataset(filename_)

    # shapenet.sim.easy
    ds = ShapeNetDataset(args, type='sim', adv_option='easy')
    filename_ = 'data/train/shapenet.sim.easy.pkl'
    ds.save_dataset(filename_)












