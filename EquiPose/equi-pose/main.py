# from time import time
import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import wandb
# import torch
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset_parser import DatasetParser
from models import get_agent
#
from common.debugger import *
from common.train_utils import cycle
# from common.eval_utils import metric_containers
# from common.ransac import ransac_delta_pose
# from vgtk.functional import so3_mean

# new
# from new_utils import display_two_pcs
from global_info import global_info

import sys
sys.path.append("../../")
from utils_common import display_two_pcs, pos_tensor_to_o3d
from utils_common import translation_error, rotation_error, adds_error
from utils_eval import EvalData
from utils_common import analyze_registration_dataset, plot_cdf
from utils_conversion import correctEquiPoseGT, equiposeDataToStandardForm, removeTranslationShapenet
from utils_conversion import ChamferDistanceSqrt


@hydra.main(config_path="configs", config_name="pose")
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(hydra.utils.get_original_cwd())
    is_cuda = (torch.cuda.is_available() and not cfg.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    #
    infos           = global_info()
    my_dir          = infos.second_path
    project_path    = infos.project_path
    categories_id   = infos.categories_id
    categories      = infos.categories

    whole_obj = infos.whole_obj
    sym_type  = infos.sym_type
    cfg.log_dir     = infos.second_path + cfg.log_dir
    cfg.model_dir   = cfg.log_dir + '/checkpoints'
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
        os.makedirs(cfg.log_dir + '/checkpoints'
        )
    #>>>>>>>>>>>>>>>>>>>>>> create network and training agent
    if 'airplane' in cfg.category:
        cfg.r_method_type=-1 # one variant
    tr_agent = get_agent(cfg)
    if cfg.use_wandb:
        run_name = f'{cfg.exp_num}_{cfg.category}'
        wandb.init(project="equi-pose", name=run_name)
        wandb.init(config=cfg)
        wandb.watch(tr_agent.net)
    #
    # load checkpoints
    # breakpoint()
    if cfg.use_pretrain or cfg.eval:
        if cfg.ckpt:
            # loading the model you specified, relative to the root dir EquiPose
            # model_dir = os.path.join(my_dir, cfg.ckpt)
            model_dir = str(my_dir) + '/' + str(cfg.ckpt)
            tr_agent.load_ckpt('best', model_dir=model_dir)
        else:
            # tr_agent.load_ckpt('latest')
            tr_agent.load_ckpt('best')

    if cfg.verbose:
        print(OmegaConf.to_yaml(cfg))
        print(cfg.log_dir)

    #>>>>>>>>>>>>>>>>>>>> dataset <<<<<<<<<<<<<<<<<<<<#
    parser = DatasetParser(cfg)
    train_loader = parser.trainloader
    val_loader   = parser.validloader
    test_loader  = parser.validloader


    # breakpoint
    # breakpoint()

    if cfg.analyze_data:

        # for num, data in enumerate(test_loader):
        #
        #     points = data['points'].squeeze(0).T
        #     pcd = pos_tensor_to_o3d(points)
        #     diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        #     print("Object diameter: ", diameter)
        # breakpoint()

        rerr, terr = analyze_registration_dataset(test_loader.dataset, cfg.name_dset,
                                                      transform=equiposeDataToStandardForm())

        plot_cdf(data=rerr, label="rotation", filename='./data_analysis/' + str(cfg.name_dset) + "rerr_test")
        plot_cdf(data=terr, label="translation", filename='./data_analysis/' + str(cfg.name_dset) + "terr_test")

        # saving
        data_ = dict()
        data_["rerr"] = rerr
        data_["terr"] = terr

        df = pd.DataFrame.from_dict(data_)
        filename = './data_analysis/' + str(cfg.name_dset) + '_test.csv'
        df.to_csv(filename)

        return None

    if cfg.writer:
        if cfg.final:
            log_dir = 'runs/' + str(cfg.new_dataset)
            writer = SummaryWriter(log_dir=log_dir)
        else:
            writer = SummaryWriter()
    else:
        writer = None

    if cfg.eval:
        if cfg.pre_compute_delta:
            # save_dict = ransac_delta_pose(cfg, train_loader)
            cfg.pre_compute_delta = False

        # all_rts, file_name, mean_err, r_raw_err, t_raw_err, s_raw_err = metric_containers(cfg.exp_num, cfg)
        # infos_dict = {'basename': [], 'in': [], 'r_raw': [], 'r_gt': [], 't_gt': [], 's_gt': [], 'r_pred': [], 't_pred': [], 's_pred': []}
        # track_dict = {'rdiff': [], 'tdiff': [], 'sdiff': [], '5deg': [], '5cm': [], '5deg5cm': [], 'chamferL1': [], 'r_acc': [], 'chirality': []}
        num_iteration = 1
        angle_error_list = []
        trans_error_list = []
        adds_error_list = []
        if 'partial' not in cfg.task:
            num_iteration = 5
        for iteration in range(num_iteration):
            cfg.iteration = iteration
            print("test iteration: ", iteration)
            for num, data in tqdm(enumerate(test_loader), total=len(test_loader)):

                # input data
                # BS = data['points'].shape[0]
                # idx = data['idx']

                if 'modelnet' in cfg.name_dset and 'shapenet' not in cfg.new_dataset:
                    data = correctEquiPoseGT(data)

                    # breakpoint()
                    points = data['points'].squeeze(0).T.to(device)
                    xyz = data['xyz'].squeeze(0).T.to(device)
                    R_gt = data['R_gt'].squeeze(0).to(device)
                    t_gt = data['T'].squeeze(0).T.to(device)

                    torch.cuda.empty_cache()

                    # obtaining transformation: from implicit reconstruction, to input
                    tr_agent.eval_func(data)
                    pose_info = tr_agent.pose_info
                    R_cannon = pose_info['r_pred'].squeeze(0)
                    t_cannon = pose_info['t_pred'].T

                    # transformed_cannon = tr_agent.transformed_pts.squeeze(0).T.to('cpu')
                    # display_two_pcs(xyz, transformed_cannon)

                    R_est = R_cannon
                    t_est = t_cannon

                elif 'shapenet' in cfg.new_dataset:
                    data = removeTranslationShapenet(data)

                    # breakpoint()
                    points = data['points'].squeeze(0).T.to(device)
                    xyz = data['xyz'].squeeze(0).T.to(device)
                    R_gt = data['R_gt'].squeeze(0).to(device)
                    t_gt = data['T'].squeeze(0).T.to(device)

                    torch.cuda.empty_cache()

                    # obtaining the transformation: from implicit reconstruction, to cad model
                    data_cannon = data
                    data_cannon['xyz'] = data_cannon['points']
                    tr_agent.eval_func(data_cannon)
                    pose_info_cannon = tr_agent.pose_info
                    R_model = pose_info_cannon['r_pred'].squeeze(0)
                    t_model = pose_info_cannon['t_pred'].T

                    # transformed_cannon = tr_agent.transformed_pts.squeeze(0).T.to('cpu')
                    # display_two_pcs(points, transformed_cannon)

                    # obtaining transformation: from implicit reconstruction, to input
                    tr_agent.eval_func(data)
                    pose_info = tr_agent.pose_info
                    R_cannon = pose_info['r_pred'].squeeze(0)
                    t_cannon = pose_info['t_pred'].T

                    # transformed_cannon = tr_agent.transformed_pts.squeeze(0).T.to('cpu')
                    # display_two_pcs(xyz, transformed_cannon)

                    R_est = R_cannon @ R_model.T
                    t_est = t_cannon - R_est @ t_model

                    # display_two_pcs(R_gt @ points + t_gt, xyz)
                    # display_two_pcs(R_est @ points + t_est, xyz)

                else:
                    raise ValueError("Dataset Incorrectly Specified.")

                # breakpoint()
                r_err_ = rotation_error(R_gt, R_est)
                t_err_ = translation_error(t_gt, t_est)
                # print("Our computed rerr: ", r_err_ * 180 / np.pi)
                # print("Their computed rerr: ", tr_agent.pose_err['rdiff'])
                # print("Our computed terr: ", t_err_)
                # print("Their computed terr: ", tr_agent.pose_err['tdiff'])
                # print("translation (GT): ", t_gt)
                # print("translation (EST): ", t_est)

                # checking their evaluation error
                # print(tr_agent.pose_err)
                points = points.to(device=R_est.device)

                chamfer_dist = ChamferDistanceSqrt()
                # breakpoint()
                X1 = R_gt @ points + t_gt
                X2 = R_est @ points + t_est
                adds_err_ = chamfer_dist(X1.T.unsqueeze(0).contiguous(), X2.T.unsqueeze(0).contiguous())

                # breakpoint()
                angle_error_list.append(r_err_.item())
                trans_error_list.append(t_err_.item())
                adds_error_list.append(adds_err_.item())

                # print("Our computed adds: ", adds_err_.item())


                # pose_diff = tr_agent.pose_err
                # if pose_diff is not None:
                #     for key in ['rdiff', 'tdiff', 'sdiff']:
                #         track_dict[key] += pose_diff[key].float().cpu().numpy().tolist()
                #     # print(pose_diff['rdiff'])
                #     deg = pose_diff['rdiff'] <= 5.0
                #     if cfg.name_dset == 'nocs_real':
                #         print('we use real world t metric!!!')
                #         cm = pose_diff['tdiff'] <= 0.05/infos.nocs_real_scale_dict[cfg.category]
                #     else:
                #         cm = pose_diff['tdiff'] <= 0.05
                #     degcm = deg & cm
                #     for key, value in zip(['5deg', '5cm', '5deg5cm'], [deg, cm, degcm]):
                #         track_dict[key] += value.float().cpu().numpy().tolist()
                #
                # if tr_agent.pose_info is not None:
                #     for key, value in tr_agent.pose_info.items():
                #         infos_dict[key] += value.float().cpu().numpy().tolist()
                #     if 'xyz' in data:
                #         input_pts  = data['xyz']
                #     else:
                #         input_pts  = data['G'].ndata['x'].view(BS, -1, 3).contiguous() # B, N, 3
                #     for m in range(BS):
                #         basename   = f'{cfg.iteration}_' + data['id'][m] + f'_' + data['class'][m]
                #         infos_dict['basename'].append(basename)
                #         infos_dict['in'].append(input_pts[m].cpu().numpy())
                #
                # if 'completion' in cfg.task:
                #     track_dict['chamferL1'].append(torch.sqrt(tr_agent.recon_loss).cpu().numpy().tolist())
                #
                # tr_agent.visualize_batch(data, "test")

        # print(f'# >>>>>>>> Exp: {cfg.exp_num} for {cfg.category} <<<<<<<<<<<<<<<<<<')
        # for key, value in track_dict.items():
        #     if len(value) < 1:
        #         continue
        #     print(key, np.array(value).mean())
        #     if key == 'rdiff':
        #         print(np.median(np.array(value)))
        #     if key == 'tdiff':
        #         print(np.median(np.array(value)))
        # if cfg.save:
        #     print('--saving to ', file_name)
        #     np.save(file_name, arr={'info': infos_dict, 'err': track_dict})

        # breakpoint()
        # save evaluated data
        if writer is not None:
            writer.add_histogram('test/rotation_err', torch.tensor(angle_error_list), bins=100)
            writer.add_histogram('test/trans_err', torch.tensor(trans_error_list), bins=100)
            writer.add_histogram('test/adds_err', torch.tensor(adds_error_list), bins=100)

            data_ = EvalData()
            data_.set_adds(np.array(adds_error_list))
            data_.set_rerr(np.array(angle_error_list))
            data_.set_terr(np.array(trans_error_list))
            savefile = str(writer.log_dir) + '/' + 'eval_data.pkl'
            data_.save(savefile)

        return None

    # >>>>>>>>>>>>>>>>>>>>>>>  main training <<<<<<<<<<<<<<<<<<<<< #
    t = torch.cuda.get_device_properties(0).total_memory / 1000000000
    r = torch.cuda.memory_reserved(0) / 1000000000
    a = torch.cuda.memory_allocated(0) / 1000000000
    f = r - a  # free inside reserved
    print("Total memory available: ", t)
    print("Total memory reserved: ", r)
    print("Total memory allocated: ", a)
    print("Free inside reserved: ", f)
    # breakpoint()
    clock = tr_agent.clock #
    val_loader  = cycle(val_loader)
    best_5deg   = 0
    best_chamferL1 = 100
    for e in range(clock.epoch, cfg.nr_epochs):
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            torch.cuda.empty_cache()

            #Test
            # mem_params = sum([param.nelement() * param.element_size() for param in tr_agent.net.parameters()])
            # mem_bufs = sum([buf.nelement() * buf.element_size() for buf in tr_agent.net.buffers()])
            # mem = mem_params + mem_bufs  # in bytes
            # mem = mem / 1000000000
            # print("Total memory occupied by the model: ", mem)
            #
            # breakpoint()
            tr_agent.train_func(data)
            # visualize
            if cfg.vis and clock.step % cfg.vis_frequency == 0:
                tr_agent.visualize_batch(data, "train")

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            infos = tr_agent.collect_loss()
            if 'r_acc' in tr_agent.infos:
                infos['r_acc'] = tr_agent.infos['r_acc']
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in infos.items()}))

            # validation step
            if clock.step % cfg.val_frequency == 0:
                torch.cuda.empty_cache()
                data = next(val_loader)
                tr_agent.val_func(data)

                if cfg.vis and clock.step % cfg.vis_frequency == 0:
                    tr_agent.visualize_batch(data, "validation")

            if clock.step % cfg.eval_frequency == 0:
                track_dict = {'rdiff': [], 'tdiff': [], 'sdiff': [],
                              '5deg': [], '5cm': [], '5deg5cm': [], 'chamferL1': [], 'r_acc': [],
                              'class_acc': []}

                for num, test_data in enumerate(test_loader):
                    if num > 100:
                        break
                    tr_agent.eval_func(test_data)
                    pose_diff = tr_agent.pose_err
                    if pose_diff is not None:
                        for key in ['rdiff', 'tdiff', 'sdiff']:
                            track_dict[key].append(pose_diff[key].cpu().numpy().mean())
                        pose_diff['rdiff'][pose_diff['rdiff']>170] = 180 - pose_diff['rdiff'][pose_diff['rdiff']>170]
                        deg = pose_diff['rdiff'] <= 5.0
                        cm = pose_diff['tdiff'] <= 0.05
                        degcm = deg & cm
                        for key, value in zip(['5deg', '5cm', '5deg5cm'], [deg, cm, degcm]):
                            track_dict[key].append(value.float().cpu().numpy().mean())
                    if 'so3' in cfg.encoder_type:
                        test_infos = tr_agent.infos
                        if 'r_acc' in test_infos:
                            track_dict['r_acc'].append(test_infos['r_acc'].float().cpu().numpy().mean())
                    if 'completion' in cfg.task:
                        track_dict['chamferL1'].append(tr_agent.recon_loss.cpu().numpy().mean())

                if cfg.use_wandb:
                    for key, value in track_dict.items():
                        if len(value) < 1:
                            continue
                        wandb.log({f'test/{key}': np.array(value).mean(), 'step': clock.step})
                if np.array(track_dict['5deg']).mean() > best_5deg:
                    tr_agent.save_ckpt('best')
                    best_5deg = np.array(track_dict['5deg']).mean()

                if np.array(track_dict['chamferL1']).mean() < best_chamferL1:
                    tr_agent.save_ckpt('best_recon')
                    best_chamferL1 = np.array(track_dict['chamferL1']).mean()

            clock.tick()
            if clock.step % cfg.save_step_frequency == 0:
                tr_agent.save_ckpt('latest')

        tr_agent.update_learning_rate()
        clock.tock()

        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()
        tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
