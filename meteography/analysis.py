# -*- coding: utf-8 -*-
"""
Functions to help with the analysis of learning algorithms.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def compare_inputs(in1, in2, shape, hist_len):
    """
    Show with matplotlib the inputs as 2 rows of `hist_len` images side-by-side
    """
    images1, images2 = [], []
    img_size = np.prod(shape)
    for i in range(hist_len):
        images1.append(in1[img_size*i:img_size*(i+1)].reshape(shape))
        images2.append(in2[img_size*i:img_size*(i+1)].reshape(shape))
    img1 = np.hstack(images1)
    img2 = np.hstack(images2)
    hline = np.zeros((1, shape[1]*hist_len))
    incmp = np.vstack([img1, hline, img2])
    plt.imshow(incmp, cmap=matplotlib.cm.Greys_r)


def compare_outputs(out1, out2, shape):
    """
    Show with matplotlib the 2 outputs as images side-by-side
    """
    img1 = out1.reshape(shape)
    img2 = out2.reshape(shape)
    vline = np.zeros((shape[0], 1))
    outcmp = np.hstack([img1, vline, img2])
    plt.imshow(outcmp, cmap=matplotlib.cm.Greys_r)


def plot_learning_curve(dataset, algo, max_ex=None, nb_points=20, **kwargs):
    """
    Train `algo` with a growing number of examples of the training set of
    `dataset` and plot (using matplotlib) the training error and validation
    error over the number of examples.

    Parameters
    ----------
    dataset : meteography.dataset.DataSet
        The dataset to take the examples from. Assumed to be already split.
    algo : meteography.perceptron.Perceptron
        The neural network to train. The initial parameters of the network
        are restored after each training session.
    max_ex : int or None
        The maximum number of examples to use for training. If None, defaults
        to the number of available examples
    nb_points : int
        The number of points to plot
    **kwargs : extra keyword arguments
        The arguments to pass to the training function
    """
    Xtrain, ytrain = dataset.training_set()
    Xval, yval = dataset.validation_set()
    if max_ex is None:
        max_ex = len(Xtrain)
    train_times = []
    train_costs = []
    valid_costs = []
    mrange = range(1, max_ex, max(max_ex // nb_points, 1))
    pvalues = algo.get_params()
    for m in mrange:
        X = Xtrain[:m]
        y = ytrain[:m]
        costs, elapsed = algo.train(X, y, **kwargs)
        train_costs.append(costs[-1])
        train_times.append(elapsed)
        valid_costs.append(algo.compute_loss(Xval, yval))
        algo.set_params(pvalues)
    #Plot the curves
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Number of examples')
    ax1.set_ylabel('Cost')
    tline = ax1.plot(mrange, train_costs, 'g', label='train. cost')
    vline = ax1.plot(mrange, valid_costs, 'r', label='valid. cost')
    ax1.legend()
    return [tline, vline]
