""" part of source code from PointNetLK (https://github.com/hmgoforth/PointNetLK), modified. """

import argparse
import os
import logging
import torch
import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter

import data_utils
import trainer
import modelnet
# import shapenet_old as shapenet
import shapenet
from train_config import options
from utils_common import display_two_pcs
from utils_common import analyze_registration_dataset, plot_cdf
import pandas as pd

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def train(args, trainset, evalset, dptnetlk):
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
                ds_name_ =  ds_name.split('_')[0] + " " + ds_name.split('_')[2]
                ds_name = ds_name_
        # breakpoint()
        rerr, terr = analyze_registration_dataset(trainset, ds_name=ds_name)

        plot_cdf(data=rerr, label="rotation", filename=str(args.dataset_type) + "rerr_train")
        plot_cdf(data=terr, label="translation", filename=str(args.dataset_type) + "terr_train")

        # saving
        data_ = dict()
        data_["rerr"] = rerr
        data_["terr"] = terr

        df = pd.DataFrame.from_dict(data_)
        filename = './data_analysis/' + str(args.dataset_type) + '_train.csv'
        df.to_csv(filename)

    else:
        # training
        model = dptnetlk.create_model()
        # breakpoint()
        if args.pretrained:
            assert os.path.isfile(args.pretrained)
            model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))

        model.to(args.device)

        checkpoint = None
        if args.resume:
            assert os.path.isfile(args.resume)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
        print('resume epoch from {}'.format(args.start_epoch+1))
        # breakpoint()
        evalloader = torch.utils.data.DataLoader(evalset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

        min_loss = float('inf')
        min_info = float('inf')

        learnable_params = filter(lambda p: p.requires_grad, model.parameters())

        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(learnable_params, lr=args.lr, weight_decay=args.decay_rate)
        else:
            optimizer = torch.optim.SGD(learnable_params, lr=args.lr)

        if checkpoint is not None:
            min_loss = checkpoint['min_loss']
            min_info = checkpoint['min_info']
            optimizer.load_state_dict(checkpoint['optimizer'])

        # training
        # breakpoint()
        LOGGER.debug('Begin Training!')
        # breakpoint()
        for epoch in range(args.start_epoch, args.max_epochs):
            print("Training")
            running_loss, running_info = dptnetlk.train_one_epoch(
                model, trainloader, optimizer, args.device, 'train', args.data_type,
                num_random_points=args.num_random_points,
                writer=None,
                epoch=epoch)

            print("Validation")
            val_loss, val_info = dptnetlk.eval_one_epoch(
                model, evalloader, args.device, 'eval', args.data_type,
                num_random_points=args.num_random_points, writer=None, epoch=epoch)

            is_best = val_loss < min_loss
            min_loss = min(val_loss, min_loss)

            LOGGER.info('epoch, %04d, %f, %f, %f, %f', epoch + 1,
                        running_loss, val_loss, running_info, val_info)
            if writer is not None:
                writer.add_scalar('Loss/train', running_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)

            snap = {'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'min_loss': min_loss,
                    'min_info': min_info,
                    'optimizer': optimizer.state_dict(), }
            if is_best:
                torch.save(model.state_dict(), '{}_{}.pth'.format(args.outfile, 'model_best'))
            torch.save(snap, '{}_{}.pth'.format(args.outfile, 'snap_last'))


def main(args):
    trainset, evalset = get_datasets(args)
    dptnetlk = trainer.TrainerAnalyticalPointNetLK(args)
    train(args, trainset, evalset, dptnetlk)


def get_datasets(args):
    cinfo = None
    if args.categoryfile:
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                    data_utils.Mesh2Points(),\
                    data_utils.OnUnitCube(),\
                    data_utils.Resampler(args.num_points)])

        traindata = data_utils.ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo)
        evaldata = data_utils.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

        trainset = data_utils.PointRegistration(traindata, data_utils.RandomTransformSE3(args.mag))
        evalset = data_utils.PointRegistration(evaldata, data_utils.RandomTransformSE3(args.mag))

    elif args.dataset_type.split('.')[0] == 'shapenet':
        type = args.dataset_type.split('.')[1]
        adv_options = args.dataset_type.split('.')[2]
        traindata = shapenet.ShapeNet(type=type, object=args.object, length=2048,
                                      num_points=args.num_points, adv_option=adv_options)
        evaldata = shapenet.ShapeNet(type=type, object=args.object, length=512,
                                     num_points=args.num_points, adv_option=adv_options)

        from utils_conversion import convertToPNLKForm
        trainset = convertToPNLKForm(traindata)
        evalset = convertToPNLKForm(evaldata)

    # elif args.dataset_type == 'shapenet_full_easy':
    #     traindata = shapenet.ShapeNet(object=args.object,
    #                                   num_of_points=args.num_points,
    #                                   type='sim',
    #                                   dataset_len=2048)
    #     evaldata = shapenet.ShapeNet(object=args.object,
    #                                  num_of_points=args.num_points,
    #                                  type='sim',
    #                                  dataset_len=512)
    #     trainset = data_utils.PointRegistration(traindata, data_utils.RandomTransformSE3(args.mag))
    #     evalset = data_utils.PointRegistration(evaldata, data_utils.RandomTransformSE3(args.mag))
    #     # breakpoint()
    #     # data = trainset[0]
    #
    # elif args.dataset_type == 'shapenet_depth_easy':
    #     traindata = shapenet.ShapeNet(object=args.object,
    #                                   num_of_points=args.num_points,
    #                                   type='real',
    #                                   dataset_len=2048)
    #     evaldata = shapenet.ShapeNet(object=args.object,
    #                                  num_of_points=args.num_points,
    #                                  type='real',
    #                                  dataset_len=512)
    #     trainset = data_utils.PointRegistration(traindata, data_utils.RandomTransformSE3(args.mag))
    #     evalset = data_utils.PointRegistration(evaldata, data_utils.RandomTransformSE3(args.mag))
    #
    # elif args.dataset_type == 'shapenet_full':
    #     if args.object == 'all':
    #         transform = torchvision.transforms.Compose([data_utils.OnUnitCube(),
    #                                                     data_utils.Resampler(args.num_points)])
    #
    #         trainset = shapenet.ShapeNetDataset(num_points=args.num_points, type='sim',
    #                                             dataset_length=2048, transform=transform)
    #         evalset = shapenet.ShapeNetDataset(num_points=args.num_points, type='sim',
    #                                            dataset_length=512, transform=transform)
    #         # trainset = shapenet.SE3PointCloudAll(num_of_points=args.num_points, dataset_len=2048)
    #         # evalset = shapenet.SE3PointCloudAll(num_of_points=args.num_points, dataset_len=512)
    #
    #     else:
    #         transform = torchvision.transforms.Compose([data_utils.OnUnitCube(),
    #                                                     data_utils.Resampler(args.num_points)])
    #
    #         trainset = shapenet.ShapeNetObjectDataset(object=args.object,
    #                                                   num_points=args.num_points,
    #                                                   type='sim',
    #                                                   dataset_length=2048,
    #                                                   transform=transform)
    #         evalset = shapenet.ShapeNetObjectDataset(object=args.object,
    #                                                  num_points=args.num_points,
    #                                                  type='sim',
    #                                                  dataset_length=512,
    #                                                  transform=transform)
    # elif args.dataset_type == 'shapenet_depth':
    #     if args.object == 'all':
    #         transform = None
    #
    #         trainset = shapenet.ShapeNetDataset(num_points=args.num_points, type='real',
    #                                             dataset_length=2048, transform=transform)
    #         evalset = shapenet.ShapeNetDataset(num_points=args.num_points, type='real',
    #                                            dataset_length=512, transform=transform)
    #
    #     else:
    #         transform = None
    #
    #         trainset = shapenet.ShapeNetObjectDataset(object=args.object,
    #                                                   num_points=args.num_points,
    #                                                   type='real',
    #                                                   dataset_length=2048,
    #                                                   transform=transform)
    #         evalset = shapenet.ShapeNetObjectDataset(object=args.object,
    #                                                  num_points=args.num_points,
    #                                                  type='real',
    #                                                  dataset_length=512,
    #                                                  transform=transform)
    else:
        raise ValueError("dataset_type not correctly specified.")

    # TEST: ShapeNet: to check the dataset.
    # breakpoint()
    # X, Y, T = trainset[0]
    # R = T[:3, :3]
    # t = T[:3, 3:]
    # Z = R @ X.T + t - Y.T
    # print(torch.sum(Z))
    # display_two_pcs(R @ X.T + t, Y.T)

    # TEST: to check the datset.
    # for idx, data in enumerate(trainset):
    #    X, Y, T = data
    #    breakpoint()
    #    R = T[:3, :3]
    #    t = T[:3, 3:]
    #    Z = R @ X.T + t - Y.T
    #    if torch.sum(Z) > 0.001:
    #        breakpoint()
    #
    #    if idx < 10:
    #        display_two_pcs(X.T, Y.T)
    #
    # for idx, data in enumerate(evalset):
    #     X, Y, T = data
    #     R = T[:3, :3]
    #     t = T[:3, 3:]
    #     Z = R @ X.T + t - Y.T
    #     if torch.sum(Z) > 0.001:
    #         breakpoint()
    #
    # print("Test passed!")
    # breakpoint()

    return trainset, evalset


if __name__ == '__main__':
    ARGS = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)
    LOGGER.debug('Training (PID=%d), %s', os.getpid(), ARGS)

    main(ARGS)

    LOGGER.debug('Training completed! Yay~~ (PID=%d)', os.getpid())
