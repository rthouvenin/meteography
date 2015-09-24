# -*- coding: utf-8 -*-
"""
Script to experiment with the artifical dataset "bigrednight", in a manner
similar to rednight.py
The main differences are:
    - larger number of training examples
    - larger pictures
    - the amount of time between each picture is not fixed
    - the time of each picture is included in the data point
    - proper validation and test set
"""

import os.path
import pickle
import random
import time

import lasagne
import matplotlib.pyplot as plt
import numpy as np
import theano as t

##
# Data set construction
##


def get_file_time(f):
    return int(f[:1]), int(f[1:3]), int(f[3:4])


def parse_filenames(file_names):
    """
    Organises the given file names in a dictionary tree with the following
    hierarchy:
    root
     +- day
         +- hour
             +- minute
    """
    file_dict = {}
    for f in file_names:
        d, h, m = get_file_time(f)
        if d not in file_dict:
            file_dict[d] = {}
        if h not in file_dict[d]:
            file_dict[d][h] = {}
        file_dict[d][h][m] = f
    return file_dict


def pick_img(file_dict, d, h, j):
    """
    Pick a file in file_dict that is approximately j hours after d,h
    and return its time.
    This is assuming there is at least one file per hour
    """
    maxd = max(file_dict.keys())
    maxh = max(file_dict[d].keys())
    rd = (d + (h+j) // maxh) % maxd
    rh = (h + j) % maxh
    rm = random.choice(file_dict[rd][rh].keys())
    return rm + 10 * rh + 1000 * rd


def make_dataset(imgpath, hist_len=5):
    """
    Builds the data set from the images located in imgpath.
    Assumes the file names have the pattern %01d%02d%01d that represent
    respectively the "day", the "hour" of the day and the "minute" of the hour.
    But each variable can be an integer of any range.
    The number of data points is the number of pictures in imgpath and each
    data point has hist_len pictures.
    """
    file_names = os.listdir(imgpath)
    file_dict = parse_filenames(file_names)
    imgshape = plt.imread(os.path.join(imgpath, file_names[0])).shape
    imgdim = np.prod(imgshape)
    n_feat = (imgdim+1) * hist_len
    n_ex = len(file_names)
    X = np.ndarray((n_ex, n_feat), dtype=np.float32)
    y = np.ndarray((n_ex, imgdim), dtype=np.float32)
    for i, f in enumerate(file_names):
        d, h, m = get_file_time(f)
        time_y = pick_img(file_dict, d, h, 4)
        file_y = os.path.join(imgpath, '%04d.png' % time_y)
        y[i, :] = plt.imread(file_y).reshape((1, imgdim))
        filename = os.path.join(imgpath, f)
        X[i, :imgdim] = plt.imread(filename).reshape((1, imgdim))
        X[i, n_feat-5] = time_y - (m + h*10 + d*1000)
        for j in range(1, 5):
            time_j = pick_img(file_dict, d, h, j)
            file_j = os.path.join(imgpath, '%04d.png' % time_j)
            s = j * imgdim
            X[i, s:s+imgdim] = plt.imread(file_j).reshape((1, imgdim))
            X[i, n_feat-5+j] = time_y - time_j
    return X, y


def split_dataset(X, y, train=.7, valid=.15):
    """
    Shuffles and splits the data set in training, validation and test sets.
    Train, valid are the percentages to sample, test will be the remaining
    """
    size_X = len(X)
    indexes = range(size_X)
    np.random.shuffle(indexes)
    size_train = int(size_X * train)
    size_valid = int(size_X * valid)
    i_train = indexes[:size_train]
    i_valid = indexes[size_train:size_train+size_valid]
    i_test = indexes[size_train+size_valid:]
    Xtrain = X[i_train, :]
    ytrain = y[i_train, :]
    Xvalid = X[i_valid, :]
    yvalid = y[i_valid, :]
    Xtest = X[i_test, :]
    ytest = y[i_test, :]
    return {'train': (Xtrain, ytrain),
            'valid': (Xvalid, yvalid),
            'test': (Xtest, ytest)}


def get_pca_reducer(X, keep_variance):
    """
    Computes the matrix to use to apply PCA to matrix X while keeping the
    variance keep_variance.
    """
    #Applies SVD
    Sigma = np.dot(X.T, X) / len(X)
    U, S, V = np.linalg.svd(Sigma)
    #Finds the smallest dimension that keeps enough variance
    total_sum = np.sum(S)
    partial_sum = 0
    dim, var = 0, 0
    while dim < len(S) and var < keep_variance:
        partial_sum += S[dim]
        var = partial_sum / total_sum
        dim += 1
    return U[:, :dim]


def reduce_dataset(dataset, keep_variance=0.99, hist_len=5):
    """
    Tries to reduce the dimensionality of the dataset by applying PCA to the
    list of all images contained in the training set.
    The same reduction is then applied to the images of the training targets,
    as well as the images of the validation and test sets and their respective
    targets.
    Returns a dictionary similar to dataset with the reduced data, and the
    reduction matrix used with label 'reducer'.
    The reduction is applied only to the images, not the data points (that are
    made of several images)
    """
    X = dataset['train'][0]
    #Build matrix of images
    imgdim = X.shape[1] / hist_len - 1
    images = X[:, :-hist_len].reshape((len(X)*hist_len, imgdim))
    Ur = get_pca_reducer(images, keep_variance)
    if Ur.shape[1] == imgdim:
        print("Could not reduce the dimensionality of the images.")
        print("Returning initial dataset.")
        result = dataset
    else:
        result = {}
        result['reducer'] = Ur
        images_r = np.dot(images, Ur)
        #Re-stack the time features to the reduced images
        X_r = np.hstack([images_r.reshape((len(X), -1)), X[:, -hist_len:]])
        y_r = np.dot(dataset['train'][1], Ur)
        result['train'] = X_r, y_r
        for dset in ['valid', 'test']:
            X = dataset[dset][0]
            if len(X):
                imgs = X[:, :-hist_len].reshape((len(X)*hist_len, imgdim))
                imgs_r = np.dot(imgs, Ur)
                imgdata = imgs_r.reshape((len(X), -1))
                X_r = np.hstack([imgdata, X[:, -hist_len:]])
                y_r = np.dot(dataset[dset][1], Ur)
                result[dset] = X_r, y_r
            else:
                X_r = np.array((0, X_r.shape[1]))
                y_r = np.array((0, y_r.shape[1]))
                result[dset] = X_r, y_r

    return result


def recover_image(reducedset, img):
    """
    From an extract img of the reducedset (example extract or target value),
    reconstructs the original data (before reduction)
    and reshapes it to an image
    """
    Ur = reducedset['reducer']
    original = np.dot(Ur, img)
    img_width = int(np.sqrt(len(original) / 3))  # assuming square and RGB
    return original.reshape((img_width, img_width, 3))


def normalize_dataset(dataset):
    result = {}
    if 'reducer' in dataset:
        result['reducer'] = dataset['reducer']
    X = dataset['train'][0]
    maxX = np.abs(np.max(X, 0, keepdims=True))
    for setname in ['train', 'valid', 'test']:
        if len(dataset[setname][0]):
            result[setname] = (dataset[setname][0] / maxX), dataset[setname][1]
    result['normalizer'] = maxX
    return result


def load_dataset(datapath='data.pack'):
    """
    Loads a dataset object from file datapath
    """
    reducedset = {}
    with open(datapath, 'r') as f:
        reducedset = pickle.load(f)
    return reducedset


##
# Network construction
##
class Network:
    def __init__(self, imgdim, nb_units=[50], learn_rate=150, hist_len=5,
                 update_func=lasagne.updates.momentum, outdim=None):
        if outdim is None:
            outdim = imgdim
        input_var = t.tensor.fmatrix('input_var')
        target_var = t.tensor.fmatrix('target_var')
        layer = lasagne.layers.InputLayer((None, (imgdim+1)*hist_len),
                                          input_var=input_var)
        sigmoid = lasagne.nonlinearities.sigmoid
        for nu in nb_units:
            layer = lasagne.layers.DenseLayer(layer, num_units=nu,
                                              nonlinearity=sigmoid)
        self.l_out = lasagne.layers.DenseLayer(layer,
                                               num_units=outdim,
                                               nonlinearity=sigmoid)

        prediction = lasagne.layers.get_output(self.l_out)
        self._predict = t.function([input_var], prediction, name='predict')
        loss = lasagne.objectives.squared_error(prediction, target_var).mean()
        self._loss = t.function([input_var, target_var], loss, name='loss')
        params = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = update_func(loss, params, learning_rate=learn_rate)
        self._train = t.function([input_var, target_var],
                                 loss, updates=updates, name='train')

    def train(self, X, target, batch_size=None,
              min_loss_diff=.999, max_epoch=500):
        print("Starting a training run...")
        if batch_size is None:
            batch_size = len(X)
        costs = []
        start_time = time.time()
        for epoch in range(max_epoch):
            for b in range(0, len(X), batch_size):
                batch = X[b:b+batch_size]
                batch_target = target[b:b+batch_size]
                cost = self._train(batch, batch_target)
            if batch_size == len(X):
                costs.append(cost)
            else:
                costs.append(self.compute_loss(X, target))
            if epoch > 0:
                conv_sample = costs[-5:]
                maxc, minc = max(conv_sample), min(conv_sample)
                converging = minc / maxc > min_loss_diff
                if converging:
                    break
        elapsed = time.time() - start_time
        print("Total training time: %.3fs in %d epochs, cost: %f"
              % (elapsed, epoch+1, costs[-1]))
        return costs, elapsed

    def predict(self, X):
        return self._predict(X)

    def compute_loss(self, X, target):
        return self._loss(X, target)

    def get_params(self):
        return lasagne.layers.get_all_param_values(self.l_out)

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.l_out, params)


def plot_learning_curve(dataset, nn, max_ex=None, nb_points=20, max_epoch=500):
        if max_ex is None:
            max_ex = len(dataset['train'][0])
        train_times = []
        train_costs = []
        valid_costs = []
        Xval = dataset['valid'][0]
        yval = dataset['valid'][1]
        mrange = range(1, max_ex, max(max_ex // nb_points, 1))
        pvalues = nn.get_params()
        for m in mrange:
            nn.set_params(pvalues)
            X = dataset['train'][0][:m]
            y = dataset['train'][1][:m]
            costs, elapsed = nn.train(X, y, max_epoch=max_epoch)
            train_costs.append(costs[-1])
            train_times.append(elapsed)
            valid_costs.append(nn.compute_loss(Xval, yval))
        #Plot the curves
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Number of examples')
        ax1.set_ylabel('Cost')
        ax1.plot(mrange, train_costs, 'g', label='train. cost')
        ax1.plot(mrange, valid_costs, 'r', label='valid. cost')
        ax1.legend()
#        ax2 = ax1.twinx()
#        ax2.set_ylabel('Time (s)')
#        ax2.plot(mrange, train_times, 'b', label='training time')
#        ax2.legend()
        plt.show()
