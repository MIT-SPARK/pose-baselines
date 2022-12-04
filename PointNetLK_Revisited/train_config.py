import argparse


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # io settings.
    parser.add_argument('--outfile', type=str, default='./logs/2021_04_17_train_modelnet',
                        metavar='BASENAME', help='output filename (prefix)')
    parser.add_argument('--dataset_path', type=str, default='./dataset/ModelNet',
                        metavar='PATH', help='path to the input dataset')

    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', type=str,
                        choices=['modelnet', 'shapenet.sim.easy', 'shapenet.sim.medium', 'shapenet.sim.hard',
                                 'shapenet.real.easy', 'shapenet.real.medium', 'shapenet.real.hard'],
                        metavar='DATASET', help='dataset type')
    parser.add_argument('--object', default='all', type=str,
                        metavar='DATASET', help='shapenet object class name')
    parser.add_argument('--data_type', default='synthetic', type=str,
                        metavar='DATASET', help='whether data is synthetic or real')
    parser.add_argument('--categoryfile', type=str, default='./dataset/modelnet40_half1.txt',
                        metavar='PATH', help='path to the categories to be trained')
    parser.add_argument('--num_points', default=1000, type=int,
                        metavar='N', help='points in point-cloud.')
    parser.add_argument('--num_random_points', default=100, type=int,
                        metavar='N', help='number of random points to compute Jacobian.')
    parser.add_argument('--mag', default=0.8, type=float,
                        metavar='D', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')
    parser.add_argument('--sigma', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--clip', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--workers', default=12, type=int,
                        metavar='N', help='number of data loading workers')

    # settings for Embedding
    parser.add_argument('--embedding', default='pointnet',
                        type=str, help='pointnet')
    parser.add_argument('--dim_k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector')

    # settings for LK
    parser.add_argument('--max_iter', default=10, type=int,
                        metavar='N', help='max-iter on LK.')

    # settings for training.
    parser.add_argument('--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--max_epochs', default=200, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        metavar='N', help='manual epoch number')
    parser.add_argument('--optimizer', default='Adam', type=str,
                        metavar='METHOD', help='name of an optimizer')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')
    parser.add_argument('--lr', type=float, default=1e-4,
                        metavar='D', help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        metavar='D', help='decay rate of learning rate')

    # settings for log
    parser.add_argument('--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file')

    # inspecting
    parser.add_argument('--writer', default=False, type=bool)
    parser.add_argument('--analyze_data', default=False, type=bool)

    args = parser.parse_args(argv)
    return args