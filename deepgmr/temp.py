
import os
import pickle
import sys
import torch
import yaml
from pytorch3d import ops


def chamfer_loss(pc, pc_, pc_padding=None, max_loss=True):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)
    pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in
    max_loss : boolean : indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

    output:
    loss    : (B, 1)
        returns max_loss if max_loss is true
    """

    if pc_padding == None:
        batch_size, _, n = pc.shape
        device_ = pc.device

        # computes a padding by flagging zero vectors in the input point cloud.
        pc_padding = ((pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3)

    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1, return_sorted=False)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist = sq_dist.squeeze(-1)*torch.logical_not(pc_padding)
    a = torch.logical_not(pc_padding)

    if max_loss:
        loss = sq_dist.max(dim=1)[0]
    else:
        loss = sq_dist.sum(dim=1)/a.sum(dim=1)

    return loss.unsqueeze(-1)

def confidence(pc, pc_):
    """
    inputs:
    pc  : input point cloud : torch.tensor of shape (B, 3, n)
    pc_ : model point cloud : torch.tensor of shape (B, 3, m)

    output:
    confidence  : torch.tensor of shape (B, 1)
    """

    return torch.exp(-chamfer_loss(pc, pc_, max_loss=True))

def confidence_kp(kp, kp_):
    """
    inputs:
    kp  : input point cloud : torch.tensor of shape (B, 3, n)
    kp_ : model point cloud : torch.tensor of shape (B, 3, m)

    output:
    confidence  : torch.tensor of shape (B, 1)

    """

    return torch.exp(-((kp-kp_)**2).sum(dim=1).max(dim=1)[0].unsqueeze(-1))


# self-supervised training and validation losses
def certify(input_point_cloud, predicted_point_cloud, corrected_keypoints,
            predicted_model_keypoints, epsilon=0.99):
    """
    inputs:
    input_point_cloud           : torch.tensor of shape (B, 3, m)
    predicted_point_cloud       : torch.tensor of shape (B, 3, n)
    corrected_keypoints         : torch.tensor of shape (B, 3, N)
    predicted_model_keypoints   : torch.tensor of shape (B, 3, N)

    outputs:
    certificate     : torch.tensor of shape (B, 1)  : dtype = torch.bool

    """

    confidence_ = confidence(input_point_cloud, predicted_point_cloud)
    confidence_kp_ = confidence_kp(corrected_keypoints, predicted_model_keypoints)

    out = (confidence_ >= epsilon) & (confidence_kp_ >= epsilon)

    return out

def chamfer_dist(pc, pc_, pc_padding=None, max_loss=False):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)
    pc_padding  : torch.tensor of shape (B, n)  : indicates if the point in pc is real-input or padded in
    max_loss : boolean : indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

    output:
    loss    : (B, 1)
        returns max_loss if max_loss is true
    """

    if pc_padding == None:
        # print(pc.shape)
        batch_size, _, n = pc.shape
        device_ = pc.device

        # computes a padding by flagging zero vectors in the input point cloud.
        pc_padding = ((pc == torch.zeros(3, 1).to(device=device_)).sum(dim=1) == 3)
        # pc_padding = torch.zeros(batch_size, n).to(device=device_)

    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist = sq_dist.squeeze(-1)*torch.logical_not(pc_padding)
    dist = torch.sqrt(sq_dist)

    a = torch.logical_not(pc_padding)

    if max_loss:
        loss = dist.max(dim=1)[0]
    else:
        loss = dist.sum(dim=1)/a.sum(dim=1)

    return loss.unsqueeze(-1)

# ADD-S and ADD-S (AUC)
def VOCap(rec, threshold):
    device_ = rec.device

    rec = torch.sort(rec)[0]
    rec = torch.where(rec <= threshold, rec, torch.tensor([float("inf")]).to(device=device_))

    n = rec.shape[0]
    prec = torch.cumsum(torch.ones(n)/n, dim=0)

    index = torch.isfinite(rec)
    rec = rec[index]
    prec = prec[index]
    # print(prec)
    # print(prec.shape)
    if rec.nelement() == 0:
        ap = torch.zeros(1)
    else:
        mrec = torch.zeros(rec.shape[0] + 2)
        mrec[0] = 0
        mrec[-1] = threshold
        mrec[1:-1] = rec

        mpre = torch.zeros(prec.shape[0]+2)
        mpre[1:-1] = prec
        mpre[-1] = prec[-1]

        for i in range(1, mpre.shape[0]):
            mpre[i] = max(mpre[i], mpre[i-1])

        ap = 0
        ap = torch.zeros(1)
        for i in range(mrec.shape[0]-1):
            # print("mrec[i+1] ", mrec[i+1])
            # print("mpre[i+1] ", mpre[i+1])
            ap += (mrec[i+1] - mrec[i]) * mpre[i+1] * (1/threshold)

    return ap


def add_s_error(predicted_point_cloud, ground_truth_point_cloud, threshold, certi=None, degeneracy_i=None, degenerate=False):
    """
    predicted_point_cloud       : torch.tensor of shape (B, 3, m)
    ground_truth_point_cloud    : torch.tensor of shape (B, 3, m)

    output:
    torch.tensor(dtype=torch.bool) of shape (B, 1)
    """

    # compute the chamfer distance between the two
    d = chamfer_dist(predicted_point_cloud, ground_truth_point_cloud, max_loss=False)
    if degeneracy_i is not None:
        if not degenerate:
            degeneracy_i = degeneracy_i > 0
        else:
            degeneracy_i = degeneracy_i < 1

    if certi is None:
        if degeneracy_i is not None:
            d = d[degeneracy_i]
        auc = VOCap(d.squeeze(-1), threshold=threshold)
    else:
        if degeneracy_i is not None:
            d = d[degeneracy_i.squeeze()*certi.squeeze()]
        else:
            d = d[certi]
        auc = VOCap(d.squeeze(-1), threshold=threshold)

    return d <= threshold, auc


def evaluate(eval_loader, model, hyper_param, certification=True, device=None, normalize_adds=False):
    model.eval()
    """
    parameters:
    -- normalize
    -- adds_threshold
    -- adds_auc_threshold
    -- epsilon
    """

    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():

        # pc_err = 0.0
        # kp_err = 0.0
        # R_err = 0.0
        # t_err = 0.0
        adds_err = 0.0
        auc = 0.0
        # we don't care about degeneracy for noncertifiable cases
        adds_err_nondeg = 0.0
        auc_nondeg = 0.0

        # pc_err_cert = 0.0
        # kp_err_cert = 0.0
        # R_err_cert = 0.0
        # t_err_cert = 0.0
        adds_err_cert = 0.0
        auc_cert = 0.0

        adds_err_cert_nondeg = 0.0
        auc_cert_nondeg = 0.0

        adds_err_cert_deg = 0.0
        auc_cert_deg = 0.0


        num_cert = 0.0
        num_nondeg = 0
        num_cert_nondeg = 0
        num_cert_deg = 0

        if normalize_adds:
            print("normalizing adds thresholds")
            model_diameter = eval_loader.dataset._get_diameter()
            print("model diameter is", model_diameter)
            hyper_param["adds_auc_threshold"] = hyper_param["adds_auc_threshold"]*model_diameter
            print(hyper_param["adds_auc_threshold"])
            hyper_param["adds_threshold"]= hyper_param["adds_threshold"]*model_diameter
            print(hyper_param["adds_threshold"])

        for i, vdata in enumerate(eval_loader):
            input_point_cloud, keypoints_target, R_target, t_target = vdata
            input_point_cloud = input_point_cloud.to(device)
            keypoints_target = keypoints_target.to(device)
            R_target = R_target.to(device)
            t_target = t_target.to(device)
            batch_size = input_point_cloud.shape[0]

            # Make predictions for this batch
            predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted, correction, predicted_model_keypoints\
                = model(input_point_cloud)

            if certification:
                certi = certify(input_point_cloud=input_point_cloud,
                                predicted_point_cloud=predicted_point_cloud,
                                corrected_keypoints=predicted_keypoints,
                                predicted_model_keypoints=predicted_model_keypoints,
                                epsilon=hyper_param['epsilon'])

            ground_truth_point_cloud = R_target @ model.cad_models + t_target
            # adds_err_, auc_ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
            #                              threshold=hyper_param["adds_threshold"])
            adds_err_, _ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
                                       threshold=hyper_param["adds_threshold"])
            _, auc_ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
                                  threshold=hyper_param["adds_auc_threshold"])

            # error for all objects
            # pc_err += pc_err_.sum()
            # kp_err += kp_err_.sum()
            # R_err += R_err_.sum()
            # t_err += t_err_.sum()
            adds_err += adds_err_.sum()
            auc += auc_

            if certification:
                # fraction certifiable
                num_cert += certi.sum()

                # error for certifiable objects
                # pc_err_cert += (pc_err_ * certi).sum()
                # kp_err_cert += (kp_err_ * certi).sum()
                # R_err_cert += (R_err_ * certi).sum()
                # t_err_cert += (t_err_ * certi).sum()
                adds_err_cert += (adds_err_ * certi).sum()

                _, auc_cert_ = add_s_error(predicted_point_cloud, ground_truth_point_cloud,
                                          threshold=hyper_param['adds_auc_threshold'], certi=certi)
                auc_cert += auc_cert_

            del input_point_cloud, keypoints_target, R_target, t_target, \
                predicted_point_cloud, predicted_keypoints, R_predicted, t_predicted

        # avg_vloss = running_vloss / (i + 1)
        # ave_pc_err = pc_err / ((i + 1)*batch_size)
        # ave_kp_err = kp_err / ((i + 1)*batch_size)
        # ave_R_err = R_err / ((i + 1)*batch_size)
        # ave_t_err = t_err / ((i + 1)*batch_size)
        ave_adds_err = 100 * adds_err / ((i + 1) * batch_size)
        ave_auc = 100 * auc / (i + 1)

        if certification:
            # ave_pc_err_cert = pc_err_cert / num_cert
            # ave_kp_err_cert = kp_err_cert / num_cert
            # ave_R_err_cert = R_err_cert / num_cert
            # ave_t_err_cert = t_err_cert / num_cert
            ave_adds_err_cert = 100 * adds_err_cert / num_cert
            ave_auc_cert = 100 * auc_cert / (i + 1)

            fra_cert = 100 * num_cert / ((i + 1)*batch_size)

        print(">>>>>>>>>>>>>>>> EVALUATING MODEL >>>>>>>>>>>>>>>>>>>>")
        print("Evaluating performance across all objects:")
        # print("pc error: ", ave_pc_err.item())
        # print("kp error: ", ave_kp_err.item())
        # print("R error: ", ave_R_err.item())
        # print("t error: ", ave_t_err.item())
        print("ADD-S (", int(hyper_param["adds_threshold"]*100), "%): ", ave_adds_err.item())
        print("ADD-S AUC (", int(hyper_param["adds_auc_threshold"]*100), "%): ", ave_auc.item())

        print("Evaluating certification: ")
        print("epsilon parameter: ", hyper_param['epsilon'])
        print("% certifiable: ", fra_cert.item())
        print("Evaluating performance for certifiable objects: ")
        # print("pc error: ", ave_pc_err_cert.item())
        # print("kp error: ", ave_kp_err_cert.item())
        # print("R error: ", ave_R_err_cert.item())
        # print("t error: ", ave_t_err_cert.item())
        print("ADD-S (", int(hyper_param["adds_threshold"]*100), "%): ", ave_adds_err_cert.item())
        print("ADD-S AUC (", int(hyper_param["adds_auc_threshold"]*100), "%): ", ave_auc_cert.item())

    del model

    return [ave_adds_err.item(), ave_auc.item(), fra_cert.item(), ave_adds_err_cert.item(), ave_auc_cert.item()]

