import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets

from utils_common import EvalData
from shapenet import OBJECT_CATEGORIES as shapenet_objects
from ycb import OBJECT_CATEGORIES as ycb_objects

datasets = ["shapenet.real.hard", "ycb.real"]

baselines = ["deepgmr",
             "equipose",
             "fpfh",
             "pointnetlk"]

baseline_folders = ["deepgmr/runs",
                    "EquiPose/equi-pose/runs",
                    "fpfh_teaser/runs",
                    "PointNetLK_Revisited/runs"]


dd_dataset = widgets.Dropdown(
    options=["shapenet", "ycb"],
    value="shapenet",
    description="Dataset"
)


dd_object = widgets.Dropdown(
    options=shapenet_objects + ycb_objects,
    value=shapenet_objects[0],
    description="Object"
)

dd_metric = widgets.Dropdown(
    options=["adds", "rerr", "terr"],
    value="adds",
    description="Metric"
)


def extract_data(my_files, my_labels):

    labels = my_labels
    data = dict()

    for i, label in enumerate(labels):
        eval_data = EvalData()

        # print("label: ", label)
        # print("loading file: ", my_files[i])
        eval_data.load(my_files[i])

        #     print(eval_data.data["adds"])
        eval_data.complete_eval_data()
        data[label] = eval_data.data

    return data


def table(my_dataset, my_object):

    #
    if my_dataset == "shapenet":
        my_dataset = "shapenet.real.hard"

        assert my_object in shapenet_objects

    elif my_dataset == "ycb":
        my_dataset = "ycb.real"

        assert my_object in ycb_objects

    else:
        raise ValueError("my_dataset not specified correctly.")

    #
    my_baselines = []
    my_files = []

    for baseline, folder in zip(baselines, baseline_folders):
        _filename = folder + '/' + my_dataset + '.' + my_object + '/eval_data.pkl'
        if os.path.isfile(_filename):
            my_baselines.append(baseline)
            my_files.append(_filename)

    #
    data = extract_data(my_files, my_baselines)

    #
    df = pd.DataFrame(data, index=["adds_th_score", "adds_auc"])
    df = df.transpose()
    display(100 * df)

    return None


def plot(my_dataset, my_object, my_metric):

    #
    if my_dataset == "shapenet":
        my_dataset = "shapenet.real.hard"

        assert my_object in shapenet_objects

    elif my_dataset == "ycb":
        my_dataset = "ycb.real"

        assert my_object in ycb_objects

    else:
        raise ValueError("my_dataset not specified correctly.")

    assert my_metric in ["adds", "rerr", "terr"]

    #
    my_baselines = []
    my_files = []

    for baseline, folder in zip(baselines, baseline_folders):
        _filename = folder + '/' + my_dataset + '.' + my_object + '/eval_data.pkl'
        if os.path.isfile(_filename):
            my_baselines.append(baseline)
            my_files.append(_filename)

    #
    data = extract_data(my_files, my_baselines)

    if my_metric == "adds":
        plot_adds(data)
    elif my_metric == "rerr":
        plot_rerr(data)
    elif my_metric == "terr":
        plot_terr(data)
    else:
        raise ValueError("my_metric not correctly specified.")

    return None


def plot_adds(data):

    sns.set(style="darkgrid")
    adds_data = dict()
    for key in data.keys():
        df_ = pd.DataFrame(dict({key: data[key]["adds"]}))
        adds_data[key] = df_

    conca = pd.concat([adds_data[key].assign(dataset=key) for key in adds_data.keys()])

    sns.kdeplot(conca, bw_adjust=0.1, cumulative=True, common_norm=False)
    plt.xlabel('ADD-S')

    return None


def plot_rerr(data):

    sns.set(style="darkgrid")
    rerr_data = dict()
    for key in data.keys():
        df_ = pd.DataFrame(dict({key: data[key]["rerr"]}))
        rerr_data[key] = df_

    conca = pd.concat([rerr_data[key].assign(dataset=key) for key in rerr_data.keys()])

    sns.kdeplot(conca, bw_adjust=0.1, cumulative=True, common_norm=False)
    plt.xlabel('Rotation Error (axis-angle, in rad)')

    return None


def plot_terr(data):

    sns.set(style="darkgrid")
    terr_data = dict()
    for key in data.keys():
        df_ = pd.DataFrame(dict({key: data[key]["terr"]}))
        terr_data[key] = df_

    conca = pd.concat([terr_data[key].assign(dataset=key) for key in terr_data.keys()])

    sns.kdeplot(conca, bw_adjust=0.1, cumulative=True, common_norm=False)
    plt.xlabel('Translation Error')

    return None



































