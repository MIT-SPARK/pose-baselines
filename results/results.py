import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets

import sys
sys.path.append("../")
from utils_eval import EvalData

# from shapenet import OBJECT_CATEGORIES as shapenet_objects
shapenet_objects = ['airplane', 'bathtub', 'bed', 'bottle',
                    'cap', 'car', 'chair', 'guitar',
                    'helmet', 'knife', 'laptop', 'motorcycle',
                    'mug', 'skateboard', 'table', 'vessel']

# from ycb import OBJECT_CATEGORIES as ycb_objects
ycb_objects = ["001_chips_can", "002_master_chef_can", "003_cracker_box", "004_sugar_box",
               "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can", "008_pudding_box",
               "009_gelatin_box", "010_potted_meat_can", "011_banana", "019_pitcher_base",
               "021_bleach_cleanser", "035_power_drill", "036_wood_block", "037_scissors",
               "040_large_marker", "051_large_clamp", "052_extra_large_clamp", "061_foam_brick"]


ycb_object_labels = {
                    "001_chips_can": "chips can",
                    "002_master_chef_can": "master chef can",
                    "003_cracker_box": "cracker box",
                    "004_sugar_box": "sugar box",
                    "005_tomato_soup_can": "tomato soup can",
                    "006_mustard_bottle": "mustard bottle",
                    "007_tuna_fish_can": "tuna fish can",
                    "008_pudding_box": "pudding box",
                    "009_gelatin_box": "gelatin box",
                    "010_potted_meat_can": "potted meat can",
                    "011_banana": "banana",
                    "019_pitcher_base": "pitcher base",
                    "021_bleach_cleanser": "bleach cleanser",
                    "035_power_drill": "power drill",
                    "036_wood_block": "wood block",
                    "037_scissors": "scissors",
                    "040_large_marker": "large marker",
                    "051_large_clamp": "large clamp",
                    "052_extra_large_clamp": "extra large clamp",
                    "061_foam_brick": "foam brick"
}

shapenet_datasets =["shapenet.sim.easy", "shapenet.sim.hard", "shapenet.real.hard"]
ycb_datasets = ["ycb.sim", "ycb.real"]

datasets = shapenet_datasets + ycb_datasets

baselines_default = ["KeyPoSim",
                     "KeyPoSimICP",
                     "KeyPoSimRANSACICP",
                     "KeyPoSimCor",
                     "KeyPoSimCorICP",
                     "KeyPoSimCorRANSACICP",
                     "c3po",
                     "KeyPoReal"]
baselines_new = ["deepgmr",
                 "equipose",
                 "fpfh",
                 "pointnetlk"]
baselines = baselines_new + baselines_default

baseline_folders = {"deepgmr": "../deepgmr/runs",
                    "equipose": "../EquiPose/equi-pose/runs",
                    "fpfh": "../fpfh_teaser/runs",
                    "pointnetlk": "../PointNetLK_Revisited/runs"}

baseline_display_name = {
             "KeyPoSim": "KeyPo (sim)",
             "KeyPoSimICP": "KeyPo (sim) + ICP",
             "KeyPoSimRANSACICP": "KeyPo (sim) + RANSAC + ICP",
             "KeyPoSimCor": "KeyPo (sim) + Corr.",
             "KeyPoSimCorICP": "KeyPo (sim) + Corr. + ICP",
             "KeyPoSimCorRANSACICP": "KeyPo (sim) + Corr. + RANSAC + ICP",
             "c3po": "C-3PO",
             "KeyPoReal": "KeyPo (real)",
             "deepgmr": "DeepGMR",
             "equipose": "EquiPose",
             "fpfh": "FPFH + TEASER++",
             "pointnetlk": "PointNetLK"
}

sim_omit_methods = ["c3po", "KeyPoReal"]

dd_dataset = widgets.Dropdown(
    options=datasets,
    value=datasets[0],
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

slide_adds_th = widgets.FloatSlider(
    min=0.01,
    max=0.20,
    step=0.005,
    value=0.05,
    description="ADD-S Threshold"
)

slide_adds_auc_th = widgets.FloatSlider(
    min=0.01,
    max=0.20,
    step=0.005,
    value=0.10,
    description="ADD-S AUC Threshold"
)


def extract_data(my_files, my_labels, my_adds_th=0.05, my_adds_auc_th=0.10):

    labels = []
    data = dict()

    for i, label in enumerate(my_labels):
        eval_data = EvalData()

        # print("label: ", label)
        # print("loading file: ", my_files[i])
        eval_data.load(my_files[i])
        eval_data.set_adds_th(my_adds_th)
        eval_data.set_adds_auc_th(my_adds_auc_th)

        #     print(eval_data.data["adds"])
        eval_data.complete_eval_data()
        data[label] = eval_data.data
        labels.append(label)

        if label == baseline_display_name["c3po"]:

            eval_data_oc = eval_data.compute_oc()
            eval_data_oc_nd = eval_data.compute_ocnd()
            label_oc = label + " (oc=1)"
            label_oc_nd = label + " (oc=1, nd=1)"

            data[label_oc] = eval_data_oc.data
            data[label_oc_nd] = eval_data_oc_nd.data

            labels.append(label_oc)
            labels.append(label_oc_nd)

    return data


def table(my_dataset, my_object, my_adds_th, my_adds_auc_th, show_table=True):

    #
    if "shapenet" in my_dataset:
        # my_dataset = "shapenet.real.hard"
        base_folder = "../c3po/expt_shapenet"

        if my_object not in shapenet_objects:
            print("Error: Specified Object not in the Dataset.")
            return None

    elif "ycb" in my_dataset:
        # my_dataset = "ycb.real"
        base_folder = "../c3po/expt_ycb"

        if my_object not in ycb_objects:
            print("Error: Specified Object not in the Dataset.")
            return None

    else:
        raise ValueError("my_dataset not specified correctly.")

    #
    if "sim" in my_dataset:
        baselines_to_plot = [x for x in baselines if x not in sim_omit_methods]
    else:
        baselines_to_plot = baselines

    #
    my_baselines = []
    my_files = []
    for baseline in baselines_to_plot:
        if baseline in baselines_new:
            folder = baseline_folders[baseline]
            _filename = folder + '/' + my_dataset + '.' + my_object + '/eval_data.pkl'
        elif baseline in baselines_default:
            _filename = base_folder + '/eval/' + baseline + '/' + "point_transformer" + '/' + my_dataset + '/' \
                        + my_object + '/eval_data.pkl'

        if os.path.isfile(_filename):
            my_baselines.append(baseline_display_name[baseline])
            my_files.append(_filename)
        # else:
        #     print(_filename)

    #
    data = extract_data(my_files, my_baselines, my_adds_th, my_adds_auc_th)

    #
    df = pd.DataFrame(data, index=["adds_th_score", "adds_auc"])
    df = df.transpose()
    if show_table:
        display(100 * df)

    return data


def plot(my_dataset, my_object, my_metric):

    #
    if "shapenet" in my_dataset:
        # my_dataset = "shapenet.real.hard"
        base_folder = "../c3po/expt_shapenet"

        if my_object not in shapenet_objects:
            print("Error: Specified Object not in the Dataset.")
            return None

    elif "ycb" in my_dataset:
        # my_dataset = "ycb.real"
        base_folder = "../c3po/expt_ycb"

        if my_object not in ycb_objects:
            print("Error: Specified Object not in the Dataset.")
            return None

    else:
        raise ValueError("my_dataset not specified correctly.")

    #
    if "sim" in my_dataset:
        baselines_to_plot = [x for x in baselines if x not in sim_omit_methods]
    else:
        baselines_to_plot = baselines

    #
    my_baselines = []
    my_files = []
    for baseline in baselines_to_plot:
        if baseline in baselines_new:
            folder = baseline_folders[baseline]
            _filename = folder + '/' + my_dataset + '.' + my_object + '/eval_data.pkl'
        elif baseline in baselines_default:
            _filename = base_folder + '/eval/' + baseline + '/' + "point_transformer" + '/' + my_dataset + '/' \
                        + my_object + '/eval_data.pkl'

        if os.path.isfile(_filename):
            my_baselines.append(baseline_display_name[baseline])
            my_files.append(_filename)
        # else:
        #     print(_filename)

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


def table_certifiable(my_dataset, my_object):

    my_detector = "point_transformer"

    #
    if "shapenet" in my_dataset:
        # if my_dataset == "shapenet":
        base_folder = "../c3po/expt_shapenet"

        if my_object not in shapenet_objects:
            print("Error: Specified Object not in the Dataset.")
            return None

    elif "ycb" in my_dataset:
        # elif my_dataset == "ycb":
        base_folder = "../c3po/expt_ycb"

        if my_object not in ycb_objects:
            print("Error: Specified Object not in the Dataset.")
            return None

        if my_detector != "point_transformer":
            print("Error: We only trained Point Transformer on YCB, as PointNet showed "
                  "suboptimal performance on ShapeNet.")
            return None

    else:
        raise ValueError("my_dataset not specified correctly.")

    #
    # if "sim" in my_dataset:
    #     baselines_to_plot = [x for x in baselines if x not in sim_omit_methods]
    # else:
    #     baselines_to_plot = baselines
    baselines_to_plot = ['c3po']

    #
    if "real" not in my_dataset:
        print("Error: this table is only available for C-3PO on shapenet.real.hard or ycb.real")
        return None

    #
    my_baselines = []
    my_files = []

    for baseline in baselines_to_plot:
        _filename = base_folder + '/eval/' + baseline + '/' + my_detector + '/' + my_dataset + '/' \
                    + my_object + '/eval_data.pkl'

        if os.path.isfile(_filename):
            my_baselines.append(baseline_display_name[baseline])
            my_files.append(_filename)

        # else:
        #     print(_filename)

    #
    data = extract_data(my_files, my_baselines)

    oc = data[baseline_display_name['c3po']]['oc']
    nd = data[baseline_display_name['c3po']]['nd']
    oc_nd = oc * nd

    percent_all = 100
    percent_oc = 100 * oc.sum() / len(oc)
    percent_oc_nd = 100 * oc_nd.sum() / len(oc_nd)

    table_data = dict()
    table_data['all'] = {'percent': percent_all}
    table_data['oc'] = {'percent': percent_oc}
    table_data['oc + nd'] = {'percent': percent_oc_nd}

    df = pd.DataFrame(table_data, index=["percent"])
    df = df.transpose()
    display(df)

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


def save_full_table(experiment):

    if experiment == "shapenet":

        data = []
        for object_ in shapenet_objects:

            data_ = table("shapenet.real.hard", object_, 0.05, 0.10, False)

            for key, dict_ in data_.items():

                if key == "C-3PO (oc=1)":
                    continue

                ds_ = dict()
                ds_['object'] = object_
                ds_['baseline'] = key
                ds_['ADD-S'] = 100 * dict_['adds_th_score']
                ds_['ADD-S AUC'] = 100 * dict_['adds_auc']

                data.append(ds_)

        filename_ = "runs/shapenet_table_full.csv"

    elif experiment == "ycb":

        data = []
        for object_ in ycb_objects:

            data_ = table("ycb.real", object_, 0.05, 0.10, False)

            for key, dict_ in data_.items():

                if key == "C-3PO (oc=1)":
                    continue

                ds_ = dict()
                ds_['object'] = object_
                ds_['baseline'] = key
                ds_['ADD-S'] = 100 * dict_['adds_th_score']
                ds_['ADD-S AUC'] = 100 * dict_['adds_auc']

                data.append(ds_)

        filename_ = "runs/ycb_table_full.csv"

    else:
        raise ValueError("experiment not specified correctly.")

    with open(filename_, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def latex_table(experiment, baselines, objects, filename=None):

    # extract data
    if experiment == "shapenet":

        data = dict()
        data_cert = dict()
        baselines_cert = ["KeyPo (sim)", "KeyPo (sim) + Corr.", "C-3PO"]
        for object_ in objects:

            data_ = table("shapenet.real.hard", object_, 0.05, 0.10, False)
            data[object_] = dict()
            data_cert[object_] = dict()
            for key, dict_ in data_.items():

                if key == "C-3PO (oc=1)" or key not in baselines:
                    continue

                ds_ = dict()
                # ds_['object'] = object_
                data[object_][key] = {'ADD-S': 100 * dict_['adds_th_score'], 'ADD-S AUC': 100 * dict_['adds_auc']}
                # ds_['ADD-S'] = 100 * dict_['adds_th_score']
                # ds_['ADD-S AUC'] = 100 * dict_['adds_auc']

                # data[object_].append(ds_)

            for b_ in baselines_cert:
                oc = data_[b_]["oc"]
                nd = data_[b_]["nd"]
                data_cert[object_][b_] = 100 * (oc * nd).sum()/len(oc)

        if filename is None:
            filename_ = "runs/shapenet_table_full.tex"
        else:
            filename_ = "runs/" + str(filename)

    elif experiment == "ycb":

        data = dict()
        data_cert = dict()
        baselines_cert = ["KeyPo (sim)", "KeyPo (sim) + Corr.", "C-3PO"]
        for object_ in objects:

            data_ = table("ycb.real", object_, 0.05, 0.10, False)
            data[ycb_object_labels[object_]] = dict()
            data_cert[ycb_object_labels[object_]] = dict()
            for key, dict_ in data_.items():

                if key == "C-3PO (oc=1)" or key not in baselines:
                    continue

                # ds_ = dict()
                # ds_['object'] = object_
                # ds_['baseline'] = key
                data[ycb_object_labels[object_]][key] = {'ADD-S': 100 * dict_['adds_th_score'], 'ADD-S AUC': 100 * dict_['adds_auc']}
                # ds_['ADD-S'] = 100 * dict_['adds_th_score']
                # ds_['ADD-S AUC'] = 100 * dict_['adds_auc']

                # data[object_].append(ds_)

            for b_ in baselines_cert:
                oc = data_[b_]["oc"]
                nd = data_[b_]["nd"]
                data_cert[ycb_object_labels[object_]][b_] = 100 * (oc * nd).sum()/len(oc)

        if filename is None:
            filename_ = "runs/ycb_table_full.tex"
        else:
            filename_ = "runs/" + filename

    else:
        raise ValueError("experiment not specified correctly.")

    # creating and saving the latex table
    lines = []

    line_ = "\\begin{tabular}{|l|"
    for o_ in objects:
        line_ += "rr|"
    line_ += "}"
    lines.append(line_)

    line_ = "\\toprule"
    lines.append(line_)

    line_ = "ADD-S~~~~ADD-S (AUC)"
    for idx, o_ in enumerate(data.keys()):
        if idx == len(data.keys()) - 1:
            line_ += f" &\multicolumn{{2}}{{c|}}{{{o_}}} "
        else:
            line_ += f" &\multicolumn{{2}}{{c}}{{{o_}}} "
    line_ += " \\\\"
    lines.append(line_)

    line_ = "\\midrule"
    lines.append(line_)

    for b_ in baselines:
        line_ = b_
        for o_ in objects:

            if "ycb" in experiment:
                o_ = ycb_object_labels[o_]

            if b_ not in data[o_].keys():
                line_ += f" & -- &  -- "
            else:
                adds = data[o_][b_]["ADD-S"]
                auc = data[o_][b_]["ADD-S AUC"]
                line_ += f" & ${adds:.2f}$ & ${auc:.2f}$ "
        line_ += "   \\\\"
        lines.append(line_)

    line_ = "\midrule"

    lines.append(line_)

    line_ = "$\\%$ (\\ocx=1, \\ndx=1)  "

    for idx, o_ in enumerate(data.keys()):
        if idx == len(data.keys()) - 1:
            line_ += f" &\multicolumn{{2}}{{c|}}{{{o_}}}"
        else:
            line_ += f" &\multicolumn{{2}}{{c}}{{{o_}}}"

    line_ += " \\\\"
    lines.append(line_)

    line_ = "\\midrule"
    lines.append(line_)

    baselines_cert = data_cert[list(data_cert.keys())[0]].keys()
    for b_ in baselines_cert:
        line_ = b_
        for o_ in objects:

            if "ycb" in experiment:
                o_ = ycb_object_labels[o_]
            # data
            percent = data_cert[o_][b_]
            line_ += f" & \multicolumn{{2}}{{c|}}{{{percent:.2f}}}  "
        line_ += "   \\\\"
        lines.append(line_)

    line_ = "\\bottomrule"

    lines.append(line_)

    line_ = "\\end{tabular}"

    lines.append(line_)

    # saving
    with open(filename_, "w") as f:
        f.write('\n'.join(lines))

    return data, data_cert



























