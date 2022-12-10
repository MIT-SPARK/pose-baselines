import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets

from utils_common import EvalData
from shapenet import OBJECT_CATEGORIES as shapenet_objects
from ycb import OBJECT_CATEGORIES as ycb_objects
from results import extract_data, plot_adds, plot_rerr, plot_terr

datasets = ["shapenet.sim.easy",
            "shapenet.sim.hard",
            "shapenet.real.easy",
            "shapenet.real.hard",
            "ycb.easy",
            "ycb.real"]

baselines = ["deepgmr",
             "equipose",
             "fpfh",
             "pointnetlk"]

baseline_folders = ["deepgmr/runs",
                    "EquiPose/equi-pose/runs",
                    "fpfh_teaser/runs",
                    "PointNetLK_Revisited/runs"]

dd_object = widgets.Dropdown(
    options=shapenet_objects + ycb_objects,
    value=shapenet_objects[0],
    description="Object"
)

dd_baseline = widgets.Dropdown(
    options=baselines,
    value=baselines[0],
    description="Method"
)

dd_metric = widgets.Dropdown(
    options=["adds", "rerr", "terr"],
    value="adds",
    description="Metric"
)

dd_data_metric = widgets.Dropdown(
    options=["rerr", "terr"],
    value="rerr",
    description="Metric"
)


def table(my_object, my_baseline):

    assert my_baseline in baselines
    folder = baseline_folders[baselines.index(my_baseline)]

    if my_object in shapenet_objects:
        my_datasets = ["shapenet.sim.easy",
                       "shapenet.sim.hard",
                       "shapenet.real.easy",
                       "shapenet.real.hard"]
    elif my_object in ycb_objects:
        my_datasets = ["ycb.sim", "ycb.real"]
    else:
        raise ValueError("my_object not specified correctly.")

    #
    my_files = []

    for dataset in my_datasets:
        _filename = folder + '/' + dataset + '.' + my_object + '/eval_data.pkl'
        if os.path.isfile(_filename):
            my_files.append(_filename)

    #
    data = extract_data(my_files, my_datasets)

    #
    df = pd.DataFrame(data, index=["adds_th_score", "adds_auc"])
    df = df.transpose()
    display(100 * df)

    return None


def plot(my_object, my_baseline, my_metric):

    assert my_baseline in baselines
    folder = baseline_folders[baselines.index(my_baseline)]

    if my_object in shapenet_objects:
        my_datasets = ["shapenet.sim.easy",
                       "shapenet.sim.hard",
                       "shapenet.real.easy",
                       "shapenet.real.hard"]
    elif my_object in ycb_objects:
        my_datasets = ["ycb.sim", "ycb.real"]
    else:
        raise ValueError("my_object not specified correctly.")

    #
    my_files = []

    for dataset in my_datasets:
        _filename = folder + '/' + dataset + '.' + my_object + '/eval_data.pkl'
        if os.path.isfile(_filename):
            my_files.append(_filename)

    #
    data = extract_data(my_files, my_datasets)

    if my_metric == "adds":
        plot_adds(data)
    elif my_metric == "rerr":
        plot_rerr(data)
    elif my_metric == "terr":
        plot_terr(data)
    else:
        raise ValueError("my_metric not correctly specified.")

    return None


def plot_data_errors(metric="rerr"):
    assert metric in ["rerr", "terr"]

    labels = ["easy", "hard"]

    files = []
    # shapenet.sim.easy
    files.append('PointNetLK_Revisited/data_analysis/shapenet.sim.easy_test.csv')

    # shapenet.sim.hard
    files.append('PointNetLK_Revisited/data_analysis/shapenet.sim.hard_test.csv')

    data = dict()

    for idx, file in enumerate(files):
        label = labels[idx]
        df = pd.read_csv(file)
        data[label] = df.to_dict('list')

    if metric == "rerr":
        rerrs = []

        for label in labels:
            dict_ = dict()
            dict_[label] = data[label]['rerr']
            df = pd.DataFrame.from_dict(dict_)
            rerrs.append(df.assign(dataset=label))

        conca = pd.concat(rerrs)
        sns.set(style="darkgrid")
        sns.kdeplot(conca, bw_adjust=0.1, cumulative=False, common_norm=False, multiple="fill")
        plt.xlabel('Axis-angle (rad)')

    elif metric == "terr":
        terrs = []

        for label in labels:
            dict_ = dict()
            dict_[label] = data[label]['terr']
            df = pd.DataFrame.from_dict(dict_)
            terrs.append(df.assign(dataset=label))

        conca = pd.concat(terrs)
        sns.set(style="darkgrid")
        sns.kdeplot(conca, bw_adjust=0.1, cumulative=False, common_norm=False, multiple="fill")
        plt.xlabel('Axis-angle (rad)')