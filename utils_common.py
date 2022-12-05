
import open3d as o3d
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from pytorch3d import ops


def translation_error(t, t_):
    """
    inputs:
    t: torch.tensor of shape (3, 1) or (B, 3, 1)
    t_: torch.tensor of shape (3, 1) or (B, 3, 1)

    output:
    t_err: torch.tensor of shape (1, 1) or (B, 1)
    """
    if t.dim() == 2:
        err = torch.norm(t - t_, p=2)/3.0
    elif t.dim() == 3:
        err = torch.norm(t-t_, p=2, dim=1)/3.0
    else:
        raise ValueError

    return err


def rotation_error(R, R_):
    """
    inputs:
    R: torch.tensor of shape (3, 3) or (B, 3, 3)
    R_: torch.tensor of shape (3, 3) or (B, 3, 3)

    output:
    R_err: torch.tensor of shape (1, 1) or (B, 1)
    """

    if R.dim() == 2:
        err = torch.acos(0.5*(torch.trace(R.T @ R)-1))
    elif R.dim() == 3:
        error = 0.5 * (torch.einsum('bii->b', torch.transpose(R, -1, -2) @ R_) - 1).unsqueeze(-1)
        epsilon = 1e-8
        err = torch.acos(torch.clamp(error, -1 + epsilon, 1 - epsilon))
    else:
        raise ValueError

    return err


def chamfer_dist(pc, pc_, max_loss=False):
    """
    inputs:
    pc  : torch.tensor of shape (B, 3, n)
    pc_ : torch.tensor of shape (B, 3, m)
    max_loss : boolean : indicates if output loss should be maximum of the distances between pc and pc_ instead of the mean

    output:
    loss    : (B, 1)
        returns max_loss if max_loss is true
    """

    #TODO: find an alternative for pytorch3d.ops
    sq_dist, _, _ = ops.knn_points(torch.transpose(pc, -1, -2), torch.transpose(pc_, -1, -2), K=1)
    # dist (B, n, 1): distance from point in X to the nearest point in Y

    sq_dist = sq_dist.squeeze(-1)
    dist = torch.sqrt(sq_dist)
    n = pc.shape[-1]

    if max_loss:
        loss = dist.max(dim=1)[0]
    else:
        loss = dist.sum(dim=1)/n

    return loss.unsqueeze(-1)


def adds_error(templet_pc, T1, T2):
    """
    Args:
        templet_pc: torch.tensor(B, 3, m)   or (3, m)
        T1: torch.tensor(B, 4, 4)   or (4, 4)
        T2: torch.tensor(B, 4, 4)   or (4, 4)

    Returns:

    """
    if len(templet_pc.shape) == 2:
        templet_pc = templet_pc.unsqueeze(0)
        T1 = T1.unsqueeze(0)
        T2 = T2.unsqueeze(0)

    pc1 = T1[:, :3, :3] @ templet_pc + T1[:, :3, 3:]
    pc2 = T2[:, :3, :3] @ templet_pc + T2[:, :3, 3:]

    err1 = chamfer_dist(pc1, pc2)
    err2 = chamfer_dist(pc2, pc1)
    err = err1 + err2
    if len(templet_pc.shape) == 2:
        err = err.squeeze(0)

    return err


def pos_tensor_to_o3d(pos, estimate_normals=True):
    """
    inputs:
    pos: torch.tensor of shape (3, N)

    output:
    open3d PointCloud
    """
    pos_o3d = o3d.utility.Vector3dVector(pos.transpose(0, 1).to('cpu').numpy())

    object = o3d.geometry.PointCloud()
    object.points = pos_o3d
    if estimate_normals:
        object.estimate_normals()

    return object


def display_batch_pcs(batch_pc, keypoints=None):
    """
    batch_pc    : torch.tensor of shape (b, 3, n)
    keypoints   : torch.tensor of shape (k, 3)

    """
    pc_list = [batch_pc[b, ...] for b in range(batch_pc.shape[0])]

    color_list = [[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.5, 0.5, 0.0],
                  [0.0, 0.5, 0.5],
                  [0.5, 0.0, 0.5],
                  [0.4, 0.4, 0.4],
                  [1.0, 1.0, 1.0]]
    color_list_len = len(color_list)
    obj_list = []
    for idx, pc in enumerate(pc_list):
        obj = pos_tensor_to_o3d(pc)
        obj.paint_uniform_color(color_list[idx % color_list_len])
        obj_list.append(obj)

    keypoint_markers = []
    if keypoints is not None:
        for xyz in keypoints:
            new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            new_mesh.translate(xyz)
            new_mesh.paint_uniform_color([0.8, 0.0, 0.0])
            keypoint_markers.append(new_mesh)

    o3d.visualization.draw_geometries(obj_list + keypoint_markers)

    return None


def display_two_pcs(pc1, pc2):
    """
    pc1 : torch.tensor of shape (3, n)
    pc2 : torch.tensor of shape (3, m)
    """
    pc1 = pc1.detach().to('cpu')
    pc2 = pc2.detach().to('cpu')
    object1 = pos_tensor_to_o3d(pos=pc1)
    object2 = pos_tensor_to_o3d(pos=pc2)

    object1.paint_uniform_color([0.8, 0.0, 0.0])
    object2.paint_uniform_color([0.0, 0.0, 0.8])

    o3d.visualization.draw_geometries([object1, object2])

    return None


def visualize_model_n_keypoints(model_list, keypoints_xyz, camera_locations=o3d.geometry.PointCloud()):
    """
    Displays one or more models and keypoints.
    :param model_list: list of o3d Geometry objects to display
    :param keypoints_xyz: list of 3d coordinates of keypoints to visualize
    :param camera_locations: optional camera location to display
    :return: list of o3d.geometry.TriangleMesh mesh objects as keypoint markers
    """
    d = 0
    for model in model_list:
        max_bound = model.get_max_bound()
        min_bound = model.get_min_bound()
        d = max(np.linalg.norm(max_bound - min_bound, ord=2), d)

    keypoint_radius = 0.01 * d

    keypoint_markers = []
    for xyz in keypoints_xyz:
        new_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=keypoint_radius)
        new_mesh.translate(xyz)
        new_mesh.paint_uniform_color([0.8, 0.0, 0.0])
        keypoint_markers.append(new_mesh)

    camera_locations.paint_uniform_color([0.1, 0.5, 0.1])
    o3d.visualization.draw_geometries(keypoint_markers + model_list + [camera_locations])

    return keypoint_markers


def visualize_torch_model_n_keypoints(cad_models, model_keypoints):
    """
    cad_models      : torch.tensor of shape (B, 3, m)
    model_keypoints : torch.tensor of shape (B, 3, N)

    """
    batch_size = model_keypoints.shape[0]

    for b in range(batch_size):

        point_cloud = cad_models[b, ...]
        keypoints = model_keypoints[b, ...].cpu()

        point_cloud = pos_tensor_to_o3d(pos=point_cloud)
        point_cloud = point_cloud.paint_uniform_color([0.0, 0.0, 1])
        point_cloud.estimate_normals()
        keypoints = keypoints.transpose(0, 1).numpy()

        visualize_model_n_keypoints([point_cloud], keypoints_xyz=keypoints)

    return 0


def analyze_registration_dataset(ds: Dataset, ds_name: str, transform=None) -> tuple:
    """
    ds      : dataset to be analyzed
    ds_name : name of the dataset, used for printing results

    note: the ds should output _, _, T, where T is the pose
    """

    print(f"Analyzing {ds_name}")
    # breakpoint()

    angles = []
    dist = []
    len_ = len(ds)

    for i in tqdm(range(len_), total=len_):
        if transform is None:
            _, _, T = ds[i]
        else:
            _, _, T = transform(ds[i])
        R = T[:3, :3]
        t = T[:3, 3:]

        temp_ = 0.5 * (torch.trace(R) - 1)
        temp1 = torch.min(torch.tensor([temp_, 0.999]))
        temp2 = torch.max(torch.tensor([temp1, -0.999]))
        angles.append(torch.acos(temp2).item())
        dist.append(torch.norm(t).item())

    return torch.tensor(angles), torch.tensor(dist)


def plot_cdf(data, label, filename):
    """
    datapoints: torch.tensor of shape (N,)
    max_val : float

    Returns:
        plots cdf up-to maximum value of max_val
    """

    plot_data = dict()
    plot_data[f"{label}"] = data
    plot_data_ = pd.DataFrame.from_dict(plot_data)

    # sns.set(stype="darkgrid")
    sns_plot = sns.kdeplot(plot_data_, bw_adjust=0.04, cumulative=True, common_norm=False)
    # plt.show()

    fig = sns_plot.get_figure()
    fig.savefig(f"{filename}.png")
    plt.close(fig)

    return None


def get_auc(rec, threshold):

    rec = np.sort(rec)
    rec = np.where(rec <= threshold, rec, np.array([float("inf")]))

    n = rec.shape[0]
    prec = np.cumsum(np.ones(n) / n, axis=0)

    index = np.isfinite(rec)
    rec = rec[index]
    prec = prec[index]
    # print(prec)
    # print(prec.shape)
    mrec = np.zeros(rec.shape[0] + 2)
    mrec[0] = 0
    mrec[-1] = threshold
    mrec[1:-1] = rec

    mpre = np.zeros(prec.shape[0] + 2)
    mpre[1:-1] = prec
    mpre[-1] = prec[-1]

    for i in range(1, mpre.shape[0]):
        mpre[i] = max(mpre[i], mpre[i - 1])

    ap = 0
    ap = np.zeros(1)
    for i in range(mrec.shape[0] - 1):
        # print("mrec[i+1] ", mrec[i+1])
        # print("mpre[i+1] ", mpre[i+1])
        # ap += (mrec[i+1] - mrec[i]) * mpre[i+1]
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1] * (1 / threshold)

    # print(ap)
    # print(type(ap))
    return ap[0]


class EvalData:
    def __init__(self, adds_th=0.02, adds_auc_th=0.05):
        self.eval_store_metrics = ["adds", "oc", "nd", \
                                   "rerr", "terr", \
                                   "adds_oc", "adds_oc_nd", \
                                   "adds_auc", "adds_oc_auc", "adds_oc_nd_auc", \
                                   "adds_th_score", "adds_oc_th_score", "adds_oc_nd_th_score", \
                                   "adds_th", "adds_auc_th"]

        self.data = dict()
        for metric in self.eval_store_metrics:
            self.data[metric] = None

        self.n = None
        self.data['adds_th'] = adds_th
        self.data['adds_auc_th'] = adds_auc_th

    def set_adds(self, adds_):
        self.data["adds"] = adds_
        self.n = len(adds_)

    def set_oc(self, oc_):
        self.data["oc"] = oc_

    def set_nd(self, nd_):
        self.data["nd"] = nd_

    def set_rerr(self, rerr_):
        self.data["rerr"] = rerr_
        self.n = len(rerr_)

    def set_terr(self, terr_):
        self.data["terr"] = terr_
        self.n = len(terr_)

    def set_adds_th(self, th_):
        self.data["adds_th"] = th_

    def set_adds_auc_th(self, th_):
        self.data["adds_auc_th"] = th_

    def complete_eval_data(self):

        if self.n is None:
            self.n = len(self.data["adds"])

        # if oc or nd is None, we fill it with all ones
        if self.data["oc"] is None:
            self.data["oc"] = np.ones(self.n)

        if self.data["nd"] is None:
            self.data["nd"] = np.ones(self.n)

        # fill adds_oc, adds_oc_nd
        idx = np.where(self.data["oc"] == 1)
        self.data["adds_oc"] = self.data["adds"][idx]

        idx = np.where(self.data["oc"] * self.data["nd"] == 1)
        self.data["adds_oc_nd"] = self.data["adds"][idx]

        # fill adds_th_score, adds_oc_th_score, adds_oc_nd_th_score
        self.data["adds_th_score"] = (self.data["adds"] <= self.data["adds_th"]).mean()
        self.data["adds_oc_th_score"] = (self.data["adds_oc"] <= self.data["adds_th"]).mean()
        self.data["adds_oc_nd_th_score"] = (self.data["adds_oc_nd"] <= self.data["adds_th"]).mean()

        # fill adds_auc, adds_oc_auc, adds_oc_nd_auc
        self.data["adds_auc"] = get_auc(self.data["adds"], self.data["adds_auc_th"])
        self.data["adds_oc_auc"] = get_auc(self.data["adds_oc"], self.data["adds_auc_th"])
        self.data["adds_oc_nd_auc"] = get_auc(self.data["adds_oc_nd"], self.data["adds_auc_th"])

    def print(self):
        """prints out the results"""

        raise NotImplementedError

    def save(self, filename):
        """saves object as a pickle file"""
        # breakpoint()
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """load object from pickle file"""

        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
        self.data = data_dict
        # breakpoint()


if __name__ == "__main__":

    datapoints = torch.randn(1000)
    max_val = 5.5
    min_val = 0.0

    plot_cdf(datapoints, "random", "first_figure")















