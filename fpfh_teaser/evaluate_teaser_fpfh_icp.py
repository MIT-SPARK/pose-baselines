import argparse
import torch
import yaml

from teaser_fpfh_icp import TEASER_FPFH_ICP

import sys
sys.path.append('../')
from shapenet import CLASS_NAME, CLASS_ID
from utils_common import display_two_pcs
# from c3po.datasets.shapenet import CLASS_NAME, CLASS_ID, FixedDepthPC
from c3po.utils.loss_functions import certify
from c3po.utils.evaluation_metrics import evaluation_error, add_s_error




def eval_(class_id, model_id, hyper_param,
          use_corrector=False, certification=True, visualize=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_name = CLASS_NAME[class_id]
    save_folder = hyper_param['save_folder']

    # setting up the dataset and dataloader
    eval_dataset_len = hyper_param['eval_dataset_len']
    eval_batch_size = hyper_param['eval_batch_size']
    eval_dataset = FixedDepthPC(class_id=class_id, model_id=model_id,
                                n=hyper_param['num_of_points_selfsupervised'],
                                num_of_points_to_sample=hyper_param['num_of_points_to_sample'],
                                dataset_len=eval_dataset_len,
                                rotate_about_z=True)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

    # get cad models
    cad_models = eval_dataset._get_cad_models().to(torch.float).to(device=device)

    # initialize the teaser module
    teaser = TEASER_FPFH_ICP(cad_models, visualize=visualize)

    # for the data batch evaluate icp output from ICP
    pc_err = 0.0
    kp_err = 0.0
    R_err = 0.0
    t_err = 0.0
    adds_err = 0.0
    auc = 0.0

    pc_err_cert = 0.0
    kp_err_cert = 0.0
    R_err_cert = 0.0
    t_err_cert = 0.0
    adds_err_cert = 0.0
    auc_cert = 0.0

    num_cert = 0.0

    with torch.no_grad():
        for i, vdata in enumerate(eval_loader):
            input_point_cloud, keypoints_target, R_target, t_target = vdata
            input_point_cloud = input_point_cloud.to(device)
            keypoints_target = keypoints_target.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)
            batch_size = input_point_cloud.shape[0]

            # applying teaser
            R_predicted, t_predicted = teaser.forward(input_point_cloud)

            # computing predicted and gt point clouds
            predicted_point_cloud = R_predicted @ cad_models + t_predicted
            ground_truth_point_cloud = R_target @ cad_models + t_target

            # visualizing
            if visualize:
                print("Displaying input pc and model with gt transformation:")
                display_two_pcs(pc1=input_point_cloud, pc2=ground_truth_point_cloud)
                print("Displaying input pc and model with teaser transformation:")
                display_two_pcs(pc1=input_point_cloud, pc2=predicted_point_cloud)

            # certification
            if certification:
                certi = certify(input_point_cloud=input_point_cloud,
                                predicted_point_cloud=predicted_point_cloud,
                                corrected_keypoints=keypoints_target,
                                predicted_model_keypoints=keypoints_target,
                                epsilon=hyper_param['epsilon'])

            # computing errors
            pc_err_, kp_err_, R_err_, t_err_ = \
                evaluation_error(input=(input_point_cloud, keypoints_target, R_target, t_target),
                                 output=(predicted_point_cloud, keypoints_target, R_predicted, t_predicted))

            adds_err_, _ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
                                       threshold=hyper_param["adds_threshold"])
            _, auc_ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
                                  threshold=hyper_param["adds_auc_threshold"])

            # error for all objects
            pc_err += pc_err_.sum()
            kp_err += kp_err_.sum()
            R_err += R_err_.sum()
            t_err += t_err_.sum()
            adds_err += adds_err_.sum()
            auc += auc_

            if certification:
                # fraction certifiable
                num_cert += certi.sum()

                # error for certifiable objects
                pc_err_cert += (pc_err_ * certi).sum()
                kp_err_cert += (kp_err_ * certi).sum()
                R_err_cert += (R_err_ * certi).sum()
                t_err_cert += (t_err_ * certi).sum()
                adds_err_cert += (adds_err_ * certi).sum()

                _, auc_cert_ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
                                           threshold=hyper_param['adds_auc_threshold'], certi=certi)
                auc_cert += auc_cert_

            del input_point_cloud, keypoints_target, R_target, t_target, \
                predicted_point_cloud, R_predicted, t_predicted, ground_truth_point_cloud

    # avg_vloss = running_vloss / (i + 1)
    ave_pc_err = pc_err / ((i + 1) * batch_size)
    ave_kp_err = kp_err / ((i + 1) * batch_size)
    ave_R_err = R_err / ((i + 1) * batch_size)
    ave_t_err = t_err / ((i + 1) * batch_size)
    ave_adds_err = 100 * adds_err / ((i + 1) * batch_size)
    ave_auc = 100 * auc / (i + 1)

    if certification:
        ave_pc_err_cert = pc_err_cert / num_cert
        ave_kp_err_cert = kp_err_cert / num_cert
        ave_R_err_cert = R_err_cert / num_cert
        ave_t_err_cert = t_err_cert / num_cert
        ave_adds_err_cert = 100 * adds_err_cert / num_cert
        ave_auc_cert = 100 * auc_cert / (i + 1)

        fra_cert = 100 * num_cert / ((i + 1) * batch_size)

    print(">>>>>>>>>>>>>>>> EVALUATING MODEL >>>>>>>>>>>>>>>>>>>>")
    print("Evaluating performance across all objects:")
    print("pc error: ", ave_pc_err.item())
    print("kp error: ", ave_kp_err.item())
    print("R error: ", ave_R_err.item())
    print("t error: ", ave_t_err.item())
    print("ADD-S (", int(hyper_param["adds_threshold"] * 100), "%): ", ave_adds_err.item())
    print("ADD-S AUC (", int(hyper_param["adds_auc_threshold"] * 100), "%): ", ave_auc.item())

    if certification:
        print("GT-certifiable: ")
        print("Evaluating certification: ")
        print("epsilon parameter: ", hyper_param['epsilon'])
        print("% certifiable: ", fra_cert.item())
        print("Evaluating performance for certifiable objects: ")
        print("pc error: ", ave_pc_err_cert.item())
        print("kp error: ", ave_kp_err_cert.item())
        print("R error: ", ave_R_err_cert.item())
        print("t error: ", ave_t_err_cert.item())
        print("ADD-S (", int(hyper_param["adds_threshold"] * 100), "%): ", ave_adds_err_cert.item())
        print("ADD-S AUC (", int(hyper_param["adds_auc_threshold"] * 100), "%): ", ave_auc_cert.item())
        print("GT-certifiable: ")

    del eval_dataset, eval_loader
    del cad_models

    return None


def evaluate(class_name, model_id, visualize=False):

    # getting class id
    class_id = CLASS_ID[class_name]

    # getting hyper-parameters
    hyper_param_file = "../expt_shapenet/self_supervised_training.yml"
    stream = open(hyper_param_file, "r")
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
    hyper_param = hyper_param['point_transformer']  # default
    hyper_param['epsilon'] = hyper_param['epsilon'][class_name]

    # print
    print(">>"*40)
    print("Analyzing Baseline for Object: ", class_name, "; Model ID:", str(model_id))

    # running evaluation
    eval_(class_id=class_id,
          model_id=model_id,
          hyper_param=hyper_param,
          visualize=visualize)

    return None


if __name__ == "__main__":
    """
    usage: 
    >> python evaluate_teaser_fpfh_icp.py --object chair 

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--object",
                        help="The ShapeNet object's class name.",
                        type=str)
    args = parser.parse_args()

    # class name: object category in shapenet
    class_name = args.object
    only_categories = [class_name]

    stream = open("../expt_shapenet/class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)
    if class_name not in model_class_ids:
        raise Exception('Invalid object. Must be a valid ShapeNet object class name.')
    else:
        model_id = model_class_ids[class_name]

    evaluate(class_name=class_name,
             model_id=model_id,
             visualize=False)