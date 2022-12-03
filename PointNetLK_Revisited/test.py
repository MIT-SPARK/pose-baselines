""" part of source code from PointNetLK (https://github.com/hmgoforth/PointNetLK), modified. """

import argparse
import os
import logging
import torch
import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter

import data_utils
import shapenet_old as shapenet
import trainer
import modelnet
from test_config import options
from utils_common import display_two_pcs, display_batch_pcs, analyze_registration_dataset, plot_cdf
import pandas as pd

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def test(args, testset, dptnetlk):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    # writer
    if args.writer is True:
        writer = SummaryWriter()
        args.outfile = str(writer.log_dir) + '/'

    else:
        writer = False


    # analyze data
    if args.analyze_data is True:
        ds_name = str(args.dataset_type)
        if ds_name.split('_')[0] == 'shapenet':
            if len(ds_name.split('_')) == 2:
                ds_name = ds_name.split('_')[0]
            else:
                ds_name_ = ds_name.split('_')[0] + " " + ds_name.split('_')[2]
                ds_name = ds_name_
        # breakpoint()
        rerr, terr = analyze_registration_dataset(testset, ds_name=ds_name)

        plot_cdf(data=rerr, label="rotation", filename=str(args.dataset_type) + "rerr_test")
        plot_cdf(data=terr, label="translation", filename=str(args.dataset_type) + "terr_test")

        # saving
        data_ = dict()
        data_["rerr"] = rerr
        data_["terr"] = terr

        df = pd.DataFrame.from_dict(data_)
        filename = './data_analysis/' + str(args.dataset_type) + '_test.csv'
        df.to_csv(filename)

    # testing model
    else:
        model = dptnetlk.create_model()

        if args.pretrained:
            assert os.path.isfile(args.pretrained)
            model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))

        model.to(args.device)

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

        # testing
        LOGGER.debug('Begin Testing!')
        # breakpoint()
        # dptnetlk.test_one_epoch(model, testloader, args.device, 'test', args.data_type, args.vis)
        dptnetlk.test_light_one_epoch(model, testloader, args.device, 'test', args.data_type, args.vis,
                                      writer=writer, savedir=args.outfile)

    return None


def main(args):
    testset = get_datasets(args)
    dptnetlk = trainer.TrainerAnalyticalPointNetLK(args)
    test(args, testset, dptnetlk)
    return None


def get_datasets(args):
    cinfo = None
    if args.categoryfile and args.data_type=='synthetic':
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                    data_utils.Mesh2Points(),\
                    data_utils.OnUnitCube(),\
                    data_utils.Resampler(args.num_points)])

        # breakpoint()
        testdata = data_utils.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)
        testset = data_utils.PointRegistration_fixed_perturbation(testdata, args.pose_file, sigma=args.sigma,
                                                                  clip=args.clip)

    elif args.dataset_type == "modelnet_train":
        transform = torchvision.transforms.Compose([ \
            data_utils.Mesh2Points(), \
            data_utils.OnUnitCube(), \
            data_utils.Resampler(args.num_points)])

        testdata = data_utils.ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo)
        testset = data_utils.PointRegistration(testdata, data_utils.RandomTransformSE3(args.mag))

    elif args.dataset_type == 'shapenet2':
        transform = torchvision.transforms.Compose([\
                    data_utils.Mesh2Points(),\
                    data_utils.OnUnitCube()])

        testdata = data_utils.ShapeNet2(args.dataset_path, transform=transform, classinfo=cinfo)
        testset = data_utils.PointRegistration_fixed_perturbation(testdata, args.pose_file,
                                                                  sigma=args.sigma, clip=args.clip)

    elif args.dataset_type == '3dmatch':
        testset = data_utils.ThreeDMatch_Testing(args.dataset_path, args.categoryfile, args.overlap_ratio, 
                                                 args.voxel_ratio, args.voxel, args.max_voxel_points, 
                                                 args.num_voxels, args.pose_file, args.vis, args.voxel_after_transf)

    elif args.dataset_type == 'shapenet_full_easy':
        testdata = shapenet.ShapeNet(object=args.object,
                                     num_of_points=args.num_points,
                                     type='sim',
                                     dataset_len=512)
        testset = data_utils.PointRegistration(testdata, data_utils.RandomTransformSE3(args.mag))

    elif args.dataset_type == 'shapenet_depth_easiest':
        # easiest: assumes the exact knowledge of occlusion (case equal to lk paper)
        testdata = shapenet.ShapeNet(object=args.object,
                                     num_of_points=args.num_points,
                                     type='real',
                                     dataset_len=512)
        testset = data_utils.PointRegistration(testdata, data_utils.RandomTransformSE3(args.mag))

    elif args.dataset_type == 'shapenet_full':
        if args.object == 'all':
            testset = shapenet.ShapeNetDataset(num_points=args.num_points, type='sim', dataset_length=512)

        else:
            testset = shapenet.ShapeNetObjectDataset(object=args.object,
                                                     num_points=args.num_points,
                                                     type='sim',
                                                     dataset_length=512)
    elif args.dataset_type == 'shapenet_depth':
        if args.object == 'all':
            testset = shapenet.ShapeNetDataset(num_points=args.num_points,
                                               type='real',
                                               dataset_length=512)

        else:
            testset = shapenet.ShapeNetObjectDataset(object=args.object,
                                                     num_points=args.num_points,
                                                     type='real',
                                                     dataset_length=512)

    elif args.dataset_type == 'shapenet_depth_easy':
        testdata = shapenet.ShapeNet(object=args.object,
                                     num_of_points=args.num_points,
                                     type='real',
                                     dataset_len=512)
        testset = data_utils.PointRegistrationEasy(testdata, data_utils.RandomTransformSE3(args.mag))

    else:
        raise ValueError("dataset_type not correctly specified.")


    # breakpoint()
    # len_ = len(testset)
    # for i in range(len_):
    #     X, Y, T = testset[i]
    #     print(i)
    #
    # breakpoint()
    # X, Y, T = testset[0]
    # R = T[:3, :3]
    # t = T[:3, 3:]
    # display_two_pcs(R @ X.T + t, Y.T)
    # display_two_pcs(X.T, Y.T)
    # TEST: ShapeNet: to check the dataset.
    # breakpoint()
    # X, Y, _, _, R, t = testset[0]
    # Z = R @ X + t - Y

    # TEST: ModelNet: to check the datset.
    # for idx, data in enumerate(testset):
    #     breakpoint()
    #     X, Y, T = data
    #     R = T[:3, :3]
    #     t = T[:3, 3:]
    #     Z = R @ X.T + t - Y.T
    #
    #     if idx < 10:
    #         display_two_pcs(X.T, Y.T)
    #
    #     if torch.sum(Z) > 0.001:
    #         breakpoint()
    #
    # print("Test passed!")
    # breakpoint()

    # TEST: 3DMatch: to check the datset.
    # for idx, data in enumerate(testset):
    #     breakpoint()
    #     X, x, Y, y, T = data
    #     # note x, y are locations of the voxels
    #     X = torch.from_numpy(X).transpose(-1, -2)
    #     x = torch.from_numpy(x)
    #     Y = Y.transpose(-1, -2)
    #     R_ =  T[:3, :3]
    #     t_ = T[:3, 3:]
    #     R = R_.unsqueeze(0)
    #     t = t_.unsqueeze(0)
    #
    #     Z1 = R @ X + t - Y
    #     print(torch.sum(Z1))
    #     Z1 = torch.cat([R @ X + t, Y], dim=-1)
    #     z1 = torch.cat([(R_ @ x.T + t_).T, y], dim=0)
    #     display_batch_pcs(Z1, z1)
    # breakpoint()

    return testset

    
if __name__ == '__main__':
    ARGS = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)
    LOGGER.debug('Testing (PID=%d), %s', os.getpid(), ARGS)

    main(ARGS)
    LOGGER.debug('Testing completed! Hahaha~~ (PID=%d)', os.getpid())
