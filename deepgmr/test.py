'''
Copyright (c) 2020 NVIDIA
Author: Wentao Yuan
'''

import argparse
import numpy as np
import os
import torch
import math
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from tensorboardX import SummaryWriter

from data import TestData
from model import DeepGMR
from pytorch3d import ops
from utils_conversion import deepgmrDataToStandardFormat, ShapeNetDataset, YCBDataset

import sys
sys.path.append("../")
from utils_common import analyze_registration_dataset, plot_cdf
from utils_common import rotation_error, translation_error, adds_error
from utils_eval import EvalData


def evaluate(model, loader, rmse_thresh, save_results=False, results_dir=None, writer=None):
    model.eval()

    # note: not using rmse_thres, save_results (which we always do).
    # inference_time = 0
    # preprocess_time = 0

    rotation_err = []
    translation_err = []
    adds_err = []

    # start = time()
    for step, (pts1, pts2, T_gt) in enumerate(tqdm(loader, leave=False)):
        # breakpoint()
        if torch.cuda.is_available():
            pts1 = pts1.cuda()
            pts2 = pts2.cuda()
            T_gt = T_gt.cuda()
        # preprocess_time += time() - start

        # start = time()
        with torch.no_grad():
            loss, r_err, t_err, rmse = model(pts1, pts2, T_gt)
            # inference_time += time() - start
            T_est = model.T_12

        r_err = r_err * math.pi / 180

        R_gt = T_gt[:, :3, :3]
        t_gt = T_gt[:, :3, 3:]
        R_est = T_est[:, :3, :3]
        t_est = T_est[:, :3, 3:]

        r_err_ = rotation_error(R_est, R_gt).squeeze(-1)
        t_err_ = translation_error(t_est, t_gt).squeeze(-1)
        adds_ = adds_error(pts1[:, :, :3].transpose(-1, -2), T_gt, T_est).squeeze(0)

        r_err = [x.item() for x in r_err_]
        t_err = [x.item() for x in t_err_]
        adds = [x.item() for x in adds_]

        rotation_err = [*rotation_err, *r_err]
        translation_err = [*translation_err, *t_err]
        adds_err = [*adds_err, *adds]

    # breakpoint()
    if writer is not None:
        # breakpoint()
        writer.add_histogram('test/rotation_err', torch.tensor(rotation_err), bins=100)
        writer.add_histogram('test/trans_err', torch.tensor(translation_err), bins=100)
        writer.add_histogram('test/adds_err', torch.tensor(adds_err), bins=100)

        data_ = EvalData()
        data_.set_adds(np.array(adds_err))
        data_.set_rerr(np.array(rotation_err))
        data_.set_terr(np.array(translation_err))
        savefile = results_dir + '/' + 'eval_data.pkl'
        data_.save(savefile)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--data_file', type=str, default=' ')
    parser.add_argument('--results_dir', type=str, default='./log')
    parser.add_argument('--checkpoint', type=str, default='models/modelnet_noisy.pth')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--rmse_thresh', type=int, default=0.2)
    # dataset
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    # model
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--n_clusters', type=int, default=16)
    parser.add_argument('--use_rri', action='store_true')
    parser.add_argument('--use_tnet', action='store_true')
    parser.add_argument('--k', type=int, default=20)

    # new for baseline
    parser.add_argument('--type', type=str,
                        choices=['deepgmr',
                                 'shapenet.sim.easy', 'shapenet.sim.medium', 'shapenet.sim.hard',
                                 'shapenet.real.easy', 'shapenet.real.medium', 'shapenet.real.hard',
                                 'ycb.sim', 'ycb.real'],
                        default='deepgmr')
    #       'deepgmr', 'shapenet.sim', 'shapenet.real', 'ycb.sim', 'ycb.real'
    # shapenet specific
    parser.add_argument('--analyze_data', type=bool, default=False)
    parser.add_argument('--object', type=str, default='all')  # could be 'all' or any shapenet object class name
    parser.add_argument('--shapenet_ds_len', type=int, default=512) # for shapenet objects' dataset
    parser.add_argument('--final', type=bool, default=False)
    # c3po evaluation
    # parser.add_argument('--eval_normalize', type=bool, default=True)
    # parser.add_argument('--eval_adds_threshold', type=float, default=0.02)
    # parser.add_argument('--eval_adds_auc_threshold', type=float, default=0.05)
    # parser.add_argument('--cert_epsilon', type=float, default=None) # default using CERT_EPSILON
    # data analysis

    #
    args = parser.parse_args()

    model = DeepGMR(args)
    if torch.cuda.is_available():
        model.cuda()

    # breakpoint()
    if args.type.split('.')[0] == 'deepgmr':
        test_data = TestData(args.data_file, args)

    elif args.type.split('.')[0] == 'shapenet':
        type = args.type.split('.')[1]
        adv_options = args.type.split('.')[2]
        test_data = ShapeNetDataset(args=args, type=type, from_file=False,
                                   adv_option=adv_options)

    elif args.type.split('.')[0] == 'ycb':
        type = args.type.split('.')[1]
        test_data = YCBDataset(args=args, type=type, split='test', from_file=False)
    else:
        raise NotImplemented

    # breakpoint()
    if args.analyze_data:

        rerr, terr = analyze_registration_dataset(test_data,
                                                  ds_name=args.type,
                                                  transform=deepgmrDataToStandardFormat())

        plot_cdf(data=rerr, label="rotation", filename='./data_analysis/' + str(args.type) + "rerr_test")
        plot_cdf(data=terr, label="translation", filename='./data_analysis/' + str(args.type) + "terr_test")

        # saving
        data_ = dict()
        data_["rerr"] = rerr
        data_["terr"] = terr

        df = pd.DataFrame.from_dict(data_)
        filename = './data_analysis/' + str(args.type) + '_test.csv'
        df.to_csv(filename)

    else:

        if args.final:
            log_dir = "runs/" + str(args.type) + '.' + str(args.object)

        writer = SummaryWriter(log_dir=log_dir)
        args.log_dir = writer.logdir
        args.results_dir = writer.logdir

        # test_data = TestData(args.data_file, args)
        test_loader = DataLoader(test_data, batch_size=args.batch_size)

        model.load_state_dict(torch.load(args.checkpoint))
        evaluate(model, test_loader, args.rmse_thresh, args.save_results, args.results_dir, writer=writer)
























