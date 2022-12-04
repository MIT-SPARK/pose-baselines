import argparse
import os
import sys
import torch
import yaml

sys.path.append('../..')

from c3po.datasets.shapenet import CLASS_NAME, CLASS_ID, FixedDepthPC
from c3po.utils.loss_functions import certify
from c3po.utils.evaluation_metrics import evaluation_error, add_s_error
from c3po.baselines.icp import RANSACwICP, TEASERwICP, wICP
from c3po.utils.visualization_utils import display_two_pcs
from c3po.expt_shapenet.proposed_model import ProposedRegressionModel as ProposedModel


def eval_icp(class_id, model_id, detector_type, hyper_param, global_registration='ransac',
             use_corrector=False, certification=True, visualize=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_name = CLASS_NAME[class_id]
    save_folder = hyper_param['save_folder']
    best_model_save_location = save_folder + class_name + '/' + model_id + '/'
    best_pre_model_save_file = best_model_save_location + '_best_supervised_kp_' + detector_type + '.pth'

    # define dataset and dataloader
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
    model_keypoints = eval_dataset._get_model_keypoints().to(torch.float).to(device=device)

    # initialize the ICP model with the cad_models
    if global_registration == 'ransac':
        icp = RANSACwICP(cad_models=cad_models, model_keypoints=model_keypoints)
    elif global_registration == 'teaser':
        icp = TEASERwICP(cad_models=cad_models, model_keypoints=model_keypoints)
    elif global_registration == 'none':
        icp = wICP(cad_models=cad_models, model_keypoints=model_keypoints)
    else:
        raise Exception("INVALID GLOBAL REGISTRATION MODULE NAME.")

    model = ProposedModel(class_name=class_name,
                          model_keypoints=model_keypoints,
                          cad_models=cad_models,
                          keypoint_detector=detector_type,
                          correction_flag=use_corrector).to(device=device)

    if not os.path.isfile(best_pre_model_save_file):
        print("ERROR: CAN'T LOAD PRETRAINED REGRESSION MODEL, PATH DOESN'T EXIST")

    state_dict_pre = torch.load(best_pre_model_save_file, map_location=device)
    model.load_state_dict(state_dict_pre)

    num_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Number of trainable parameters in the keypoint detector: ", num_parameters)

    model.eval()
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

            _, detected_keypoints, R0, t0, _ \
                = model(input_point_cloud)

            if global_registration == 'ransac' or global_registration == 'teaser':
                predicted_point_cloud, R_predicted, t_predicted = icp.forward(input_point_cloud, detected_keypoints)
            elif global_registration == 'none':
                predicted_point_cloud, R_predicted, t_predicted = icp.forward(input_point_cloud, R0, t0)
            else:
                print("INVALID GLOBAL REGISTRATION MODULE NAME.")


            if visualize:
                pc_in = input_point_cloud[0, ...]
                pc_pred = predicted_point_cloud[0, ...]
                display_two_pcs(pc1=pc_in, pc2=pc_pred)

            if certification:
                certi = certify(input_point_cloud=input_point_cloud,
                                predicted_point_cloud=predicted_point_cloud,
                                corrected_keypoints=keypoints_target,
                                predicted_model_keypoints=keypoints_target,
                                epsilon=hyper_param['epsilon'])

            pc_err_, kp_err_, R_err_, t_err_ = \
                evaluation_error(input=(input_point_cloud, keypoints_target, R_target, t_target),
                                 output=(predicted_point_cloud, keypoints_target, R_predicted, t_predicted))

            ground_truth_point_cloud = R_target @ cad_models + t_target
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
                predicted_point_cloud, R_predicted, t_predicted, detected_keypoints, ground_truth_point_cloud, R0, t0

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

    del eval_dataset, eval_loader, model, state_dict_pre
    del cad_models, model_keypoints

    return None


def evaluate_icp(class_name, model_id, detector_type,
                 global_registration='ransac', use_corrector=False, visualize=False):

    class_id = CLASS_ID[class_name]
    hyper_param_file = "../expt_shapenet/self_supervised_training.yml"
    stream = open(hyper_param_file, "r")
    hyper_param = yaml.load(stream=stream, Loader=yaml.FullLoader)
    hyper_param = hyper_param[detector_type]   # we only use the evaluation dataset parameters, which are the same
    hyper_param['epsilon'] = hyper_param['epsilon'][class_name]

    print(">>"*40)
    print("Analyzing Baseline for Object: ", class_name, "; Model ID:", str(model_id))

    eval_icp(class_id=class_id,
             model_id=model_id,
             detector_type=detector_type,
             hyper_param=hyper_param,
             global_registration=global_registration,
             use_corrector=use_corrector,
             visualize=visualize)

    return None


if __name__ == "__main__":
    """
    usage: 
    >> python evaluate_icp.py "chair" "teaser" "nc"
    >> python evaluate_icp.py "table" "ransac" "c"
    >> python evaluate_icp.py "table" "none" "nc"

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("class_name", help="specify the ShapeNet class name.", type=str)
    parser.add_argument("global_registration", help="either ransac or teaser or none", type=str)
    parser.add_argument("corrector_flag", help="c for corrector, nc for no corrector", type=str)

    args = parser.parse_args()

    # class name: object category in shapenet
    class_name = args.class_name
    # global_registration: ransac, teaser, or none which uses registration output assuming all the detected keypoints
    global_registration = args.global_registration
    # correction flag: c = uses the corrector, nc = does not use the corrector, in the forward pass
    corrector_flag = args.corrector_flag
    if corrector_flag == 'c':
        use_corrector = True
    elif corrector_flag == 'nc':
        use_corrector = False
    else:
        raise Exception("CORRECTOR FLAG INPUT INCORRECT.")

    only_categories = [class_name]

    stream = open("../expt_shapenet/class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)
    if class_name not in model_class_ids:
        raise Exception('Invalid class_name')
    else:
        model_id = model_class_ids[class_name]

    evaluate_icp(class_name=class_name,
                 model_id=model_id,
                 detector_type='point_transformer',
                 global_registration=global_registration,
                 use_corrector=use_corrector,
                 visualize=False)