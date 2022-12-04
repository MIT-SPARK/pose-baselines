
import torch
import pickle
from tqdm import tqdm


class convertToPNLKForm(torch.utils.data.Dataset):
    """
    Wraper for shapenet datasets in shapenet.py, for DeepGMR training.

    """
    def __init__(self, dataset, from_file=None, filename=None):

        self.filename = filename
        self.from_file = from_file

        if from_file and filename is not None:
            with open(self.filename, 'rb') as f:
                self.data_ = pickle.load(f)

        else:
            self.ds = dataset

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

        T = torch.eye(4)
        T[:3, :3] = R
        T[:3, 3:] = t

        return pc0.T, pc1.T, T

    def save_dataset(self, filename):
        data_ = []
        len_ = len(self.ds)
        for i in tqdm(range(len_)):
            # breakpoint()
            data = self.ds[i]
            data_.append(data)

        with open(filename, 'wb') as f:
            pickle.dump(data_, f, protocol=pickle.HIGHEST_PROTOCOL)