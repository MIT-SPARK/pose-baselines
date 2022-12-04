# -*- coding: future_fstrings -*-
import MinkowskiEngine as ME
import numpy

#

import numpy as np
import sys
import argparse
import logging
import torch

sys.path.append("../..")

from c3po.baselines.fcgf.config import get_config
from c3po.baselines.fcgf.liby.data_loaders_shapenet import make_data_loader
from c3po.baselines.fcgf.liby.trainer import ContrastiveLossTrainer, HardestContrastiveLossTrainer, \
    TripletLossTrainer, HardestTripletLossTrainer

from c3po.baselines.fcgf.model import load_model

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")


def get_trainer(trainer):
  if trainer == 'ContrastiveLossTrainer':
    return ContrastiveLossTrainer
  elif trainer == 'HardestContrastiveLossTrainer':
    return HardestContrastiveLossTrainer
  elif trainer == 'TripletLossTrainer':
    return TripletLossTrainer
  elif trainer == 'HardestTripletLossTrainer':
    return HardestTripletLossTrainer
  else:
    raise ValueError(f'Trainer {trainer} not found')


def _temp_test(config, train_dl, val_dl):
    # DEBUGGING PART
    ds = train_dl.dataset
    (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans) = ds[0]
    breakpoint()
    num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.
    #
    # Model initialization
    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=config.bn_momentum,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3)
    dl = train_dl
    voxel_size = config.voxel_size
    for idx, data in enumerate(dl):
        # (xyz0, xyz1, coords0, coords1, feats0, feats1, matches, trans) = data
        breakpoint()
        pcd0 = data['pcd0']
        pcd1 = data['pcd1']
        sinput0_C = data['sinput0_C']
        sinput0_F = data['sinput0_F']
        sinput1_C = data['sinput1_C']
        sinput1_F = data['sinput1_F']
        pos_pairs = data['correspondences']
        T_gt = data['T_gt']
        len_batch = data['len_batch']

        sinput0 = ME.SparseTensor(coordinates=sinput0_C, features=sinput0_F)
        sinput1 = ME.SparseTensor(coordinates=sinput1_C, features=sinput1_F)
        F0 = model(sinput0).F
        F1 = model(sinput1).F

        idx0 = pos_pairs[:, 0].long()
        idx1 = pos_pairs[:, 1].long()

        breakpoint()
        len0 = pcd0.shape[0]
        len1 = pcd1.shape[0]

        len0_ = F0.shape[0]
        len1_ = F1.shape[0]

        if torch.any(idx0 >= len0) or torch.any(idx1 >= len1):
            print("A")
            breakpoint()

        if torch.any(idx0 >= len0_) or torch.any(idx1 >= len1_):
            print("B")
            breakpoint()

        if (len0 != len0_):
            print("len0 vs len0_: ", len0 - len0_)

        if (len1 != len1_):
            print("len1 vs len1_: ", len1 - len1_)

    return None


def main(config):

    # config.dataset_length = 2048
    train_dl = make_data_loader(type=config.type,
                                dataset_length=config.train_data_len_shapenet,
                                batch_size=config.batch_size,
                                voxel_size=config.voxel_size,
                                positive_pair_search_voxel_size_multiplier=config.positive_pair_search_voxel_size_multiplier
                                )

    # config.dataset_length = 512
    val_dl = make_data_loader(type=config.type,
                              dataset_length=config.val_data_len_shapenet,
                              batch_size=config.val_batch_size,
                              voxel_size=config.voxel_size,
                              positive_pair_search_voxel_size_multiplier=config.positive_pair_search_voxel_size_multiplier
                              )

    # _temp_test(config, train_dl, val_dl)
    # breakpoint()
    Trainer = get_trainer(config.trainer)
    trainer = Trainer(
      config=config,
      data_loader=train_dl,
      val_data_loader=val_dl,
    )

    train_loss_per_epoch = trainer.train()
    with open(config.out_dir + '/train_loss_per_epoch.npz', 'wb') as f:
        np.save(f, np.asarray(train_loss_per_epoch))


if __name__ == "__main__":
    CUDA_LAUNCH_BLOCKING=1
    config = get_config()
    config.out_dir = config.out_dir + "/" + str(config.type)
    # config.type = 'sim' # 'real'
    # config.max_epoch = 2
    # config.voxel_size = 0.025
    # config.batch_size = 4
    # config.positive_pair_search_voxel_size_multiplier = 1.5
    # config.dataset_length = 2000

    main(config)
