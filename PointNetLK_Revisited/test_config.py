import argparse


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # io settings.
    parser.add_argument('--outfile', type=str, default='./test_logs/2021_04_17_test_on_3dmatch_trained_on_modelnet',
                        metavar='BASENAME', help='output filename (prefix)')
    parser.add_argument('--dataset_path', type=str, default='./dataset/ThreeDMatch',
                        metavar='PATH', help='path to the input dataset')
    parser.add_argument('--categoryfile', type=str, default='./dataset/test_3dmatch.txt',
                        metavar='PATH', choices=['./dataset/test_3dmatch.txt', './dataset/modelnet40_half2.txt', './dataset/modelnet40_half1.txt'],
                        help='path to the categories to be tested')
    parser.add_argument('--pose_file', type=str, default='./dataset/gt_poses.csv',
                        metavar='PATH', help='path to the testing pose files')

    # settings for input data
    parser.add_argument('--dataset_type', default='3dmatch', type=str,
                        metavar='DATASET',
                        choices=['modelnet', '3dmatch', 'modelnet_train',
                                 'shapenet.sim.easy', 'shapenet.sim.medium', 'shapenet.sim.hard',
                                 'shapenet.real.easy', 'shapenet.real.medium', 'shapenet.real.hard'],
                        help='dataset type')
    parser.add_argument('--data_type', default='synthetic', type=str,
                        metavar='DATASET', help='whether data is synthetic or real')    #real means 3dmatch.
    parser.add_argument('--object', default='all', type=str,
                        metavar='DATASET', help='shapenet object class name')
    parser.add_argument('--num_points', default=1000, type=int,
                        metavar='N', help='points in point-cloud')
    parser.add_argument('--sigma', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--clip', default=0.00, type=float,
                        metavar='D', help='noise range in the data')
    parser.add_argument('--workers', default=12, type=int,
                        metavar='N', help='number of data loading workers')

    # settings for voxelization
    parser.add_argument('--overlap_ratio', default=0.7, type=float,
                        metavar='D', help='overlapping ratio for 3DMatch dataset.')
    parser.add_argument('--voxel_ratio', default=0.05, type=float,
                        metavar='D', help='voxel ratio')
    parser.add_argument('--voxel', default=2, type=float,
                        metavar='D', help='how many voxels you want to divide in each axis')
    parser.add_argument('--max_voxel_points', default=1000, type=int,
                        metavar='N', help='maximum points allowed in a voxel')
    parser.add_argument('--num_voxels', default=8, type=int,
                        metavar='N', help='number of voxels')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='whether to visualize or not')
    parser.add_argument('--voxel_after_transf', action='store_true', default=False,
                        help='given voxelization before or after transformation')

    # settings for Embedding
    parser.add_argument('--embedding', default='pointnet',
                        type=str, help='pointnet')
    parser.add_argument('--dim_k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector')

    # settings for LK
    parser.add_argument('-mi', '--max_iter', default=20, type=int,
                        metavar='N', help='max-iter on LK.')

    # settings for training.
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    # settings for log
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile')
    parser.add_argument('--pretrained', default='./logs/model_trained_on_ModelNet40_model_best.pth', type=str,
                        metavar='PATH', help='path to pretrained model file ')

    # imported from train_config
    parser.add_argument('--mag', default=0.8, type=float,
                        metavar='D', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')

    # inspecting
    parser.add_argument('--writer', default=False, type=bool)
    parser.add_argument('--analyze_data', default=False, type=bool)

    args = parser.parse_args(argv)
    return args