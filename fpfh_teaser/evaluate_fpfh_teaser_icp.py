import argparse
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from teaser_fpfh_icp import TEASER_FPFH_ICP

import sys
sys.path.append('../')
from shapenet import ShapeNet
from utils_common import display_two_pcs
from utils_common import adds_error, rotation_error, translation_error
from utils_common import EvalData


def evaluate(dataset_name, object, visualize=False):

    device = 'cpu'
    writer = SummaryWriter()
    savedir = str(writer.log_dir) + '/'

    if dataset_name.split('.')[0] == 'shapenet':
        type = dataset_name.split('.')[1]
        adv_option = dataset_name.split('.')[2]
        eval_dataset = ShapeNet(type=type, object=object, length=512, num_points=1024, adv_option=adv_option)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False)
    else:
        raise NotImplementedError

    # breakpoint()
    if object != 'all':
        # get cad models
        cad_models = eval_dataset._get_cad_models().to(torch.float).to(device=device)

        # initialize the teaser module
        teaser = TEASER_FPFH_ICP(cad_models, visualize=False)

    #
    adds_err = []
    R_err = []
    t_err = []

    with torch.no_grad():
        for i, vdata in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            pc0, pc1, _, _, R_target, t_target = vdata
            pc0 = pc0.to(device)
            pc1 = pc1.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)

            input_point_cloud = pc1
            #
            if object == 'all':
                # assuming: pc0 is of shape (1, 3, n)
                cad_models = pc0
                teaser = TEASER_FPFH_ICP(cad_models, visualize=False)

            # applying teaser
            R_predicted, t_predicted = teaser.forward(input_point_cloud)

            # visualizing
            if visualize:
                # computing predicted and gt point clouds
                predicted_point_cloud = R_predicted @ cad_models + t_predicted
                ground_truth_point_cloud = pc0

                print("Displaying input pc and model with gt transformation:")
                display_two_pcs(pc1=input_point_cloud.squeeze(0), pc2=ground_truth_point_cloud.squeeze(0))
                print("Displaying input pc and model with teaser transformation:")
                display_two_pcs(pc1=input_point_cloud.squeeze(0), pc2=predicted_point_cloud.squeeze(0))

            T_pred = torch.eye(4).to(device)
            T_pred[:3, :3] = R_predicted.squeeze(0)
            T_pred[:3, 3:] = t_predicted.squeeze(0)

            T_gt = torch.eye(4).to(device)
            T_gt[:3, :3] = R_target.squeeze(0)
            T_gt[:3, 3:] = t_target.squeeze(0)

            # breakpoint()
            adds_err_ = adds_error(pc0.squeeze(0), T_pred, T_gt).to('cpu')
            r_err_ = rotation_error(R_target, R_predicted).to('cpu')
            t_err_ = translation_error(t_target, t_predicted).to('cpu')

            adds_err.append(float(adds_err_.item()))
            R_err.append(float(r_err_.item()))
            t_err.append(float(t_err_.item()))

    # breakpoint()

    eval_data = EvalData()
    import numpy as np
    eval_data.set_adds(np.asarray(adds_err))
    eval_data.set_rerr(np.asarray(R_err))
    eval_data.set_terr(np.asarray(t_err))

    savefile = savedir + 'eval_data.pkl'
    eval_data.save(savefile)

    return None


if __name__ == "__main__":
    """
    usage: 
    >> python evaluate_teaser_fpfh_icp.py --dataset shapenet.sim.easy --object chair 

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--object",
                        help="The ShapeNet/YCB object's class name.",
                        type=str)
    parser.add_argument('--dataset',
                        choices=['ycb',
                                 'shapenet.sim.easy', 'shapenet.sim.medium', 'shapenet.sim.hard',
                                 'shapenet.real.easy', 'shapenet.real.medium', 'shapenet.real.hard'],
                        help="Dataset name",
                        type=str)
    args = parser.parse_args()

    # class name: object category in shapenet
    object = args.object
    dataset_name = args.dataset

    evaluate(dataset_name=dataset_name, object=object, visualize=False)