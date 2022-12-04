#TODO: Write a code that:
# 1. extracts the numpy file, which stores all the train loss,
# 2. extracts all checkpoint.pth, evaluates them with ADD-S and ADD-S AUC, and
# 3. plots both as a function of number of epochs.
# -- care needs to be taken for the zeroth epoch. We need to compute the ADD-S and ADD-S AUC for the pre-trained model

import argparse

import numpy
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from evaluate_teaser_fcgf_icp import evaluate


def _make_plot(adds, adds_auc, train_loss):

    num_epochs = len(train_loss)
    epochs = torch.arange(1, num_epochs+1)
    x = adds
    z = adds_auc
    y = train_loss

    # breakpoint()
    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()
    # make a plot
    ax.plot(epochs,
            x,
            color="red",
            marker="o")
    ax.plot(epochs, z, color="green", marker="+")
    # set x-axis label
    ax.set_xlabel("Training epoch", fontsize=14)
    # set y-axis label
    ax.set_ylabel("ADD-S, ADD-S AUC",
                  color="black",
                  fontsize=14)


    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(epochs, y, color="blue", marker="o")
    ax2.set_ylabel("Training Loss", color="blue", fontsize=14)
    plt.show()

    # save the plot as a file
    # fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
    #             format='jpeg',
    #             dpi=100,
    #             bbox_inches='tight')


def plot(args):

    model_dir = str(args.model_dir)
    train_loss_per_epoch = np.load(model_dir + '/train_loss_per_epoch.npz')

    if args.epochs is None:
        num_epochs = len(train_loss_per_epoch)
    else:
        num_epochs = int(args.epochs)
        train_loss_per_epoch = train_loss_per_epoch[:num_epochs]

    stream = open("../expt_shapenet/class_model_ids.yml", "r")
    model_class_ids = yaml.load(stream=stream, Loader=yaml.Loader)
    if args.object == 'all':
        class_name_list = model_class_ids.keys()
    else:
        class_name_list = [str(args.object)]

    adds = np.zeros(num_epochs)
    adds_auc = np.zeros(num_epochs)
    # breakpoint()
    for class_name in class_name_list:

        model_id = model_class_ids[class_name]
        adds_ = []
        adds_auc_ = []

        for epoch in range(1, num_epochs+1):

            model = model_dir + f'/checkpoint-{epoch}.pth'

            out_dict = evaluate(class_name=class_name,
                                model_id=model_id,
                                model=model,
                                pre_trained_3dmatch=True,
                                visualize=False)

            adds_.append(out_dict['ADD-S'])
            adds_auc_.append(out_dict['ADD-S AUC'])

        adds_ = numpy.asarray(adds_)
        adds_auc_ = numpy.asarray(adds_auc_)

        adds = adds + adds_
        adds_auc = adds_auc + adds_auc_

    num_classes = len(class_name_list)
    adds = adds / num_classes
    adds_auc = adds_auc / num_classes

    # breakpoint()
    _make_plot(adds=adds, adds_auc=adds_auc, train_loss=train_loss_per_epoch)

    return None


if __name__ == "__main__":
    """
    usage: 
    >> python teaser_fcgf_icp_plot_loss.py --object 'chair' --model_dir <dir-name> --epochs 5
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--object",
                        help="ShapeNet object class name.",
                        type=str,
                        default='all')
    parser.add_argument("--model_dir",
                        help="Specify the folder that contains all checkpoint model files (.pth)",
                        type=str,
                        default=None)
    parser.add_argument("--epochs",
                        type=int,
                        default=None)
    args = parser.parse_args()

    plot(args)

