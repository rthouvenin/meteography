# -*- coding: utf-8 -*-

import os
from os import path
import time

import lasagne
import matplotlib.pyplot as plt
import numpy as np
import theano as t

imgpath = 'artificial/rednight/'
files = [int(f[:-4]) for f in os.listdir(imgpath)]
files.sort()

def img_file_name(idimg):
    return path.join(imgpath, str(files[idimg]) + '.png')

history_length = 5
imgshape = plt.imread(img_file_name(0)).shape
imgdim = np.prod(imgshape)
n = imgdim * history_length + 1
m = len(files)

#prepares input and target data
X = np.ndarray((m, n), dtype=np.float32) 
y = np.ndarray((m, imgdim), dtype=np.float32)
for i in range(m):
    X[i, n-1] = float(i) / m
    for imgid in range(history_length):
        nstart = imgid * imgdim
        nend = (imgid + 1) * imgdim
        f = img_file_name((i + imgid) % m)
        X[i, nstart:nend] = plt.imread(f).reshape((1, imgdim))
    f = img_file_name((i + history_length) % m)
    y[i, :] = plt.imread(f).reshape((1, imgdim))

#Constructs neural network
input_var = t.tensor.fmatrix('input_var')
target_var = t.tensor.fmatrix('target_var')

l_in = lasagne.layers.InputLayer(X.shape, input_var=input_var)
l_h = lasagne.layers.DenseLayer(l_in, num_units=100, 
                    nonlinearity=lasagne.nonlinearities.sigmoid)
l_out = lasagne.layers.DenseLayer(l_h, num_units=imgdim,
                    nonlinearity=lasagne.nonlinearities.sigmoid)


prediction = lasagne.layers.get_output(l_out)
loss = lasagne.objectives.squared_error(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.sgd(loss, params, learning_rate=100)

train = t.function([input_var, target_var], loss, updates=updates, name='train')

def train_batch(nb_epoch):
    print("Starting a training batch...")
    costs = np.ndarray(nb_epoch)
    start_time = time.time()
    for epoch in range(nb_epoch):
        epoch_time = time.time()
        costs[epoch] = train(X, y)
        epoch_time = time.time() - epoch_time
        print("Epoch %d done in %.3fs, loss: %.5f" % (epoch, epoch_time, costs[epoch]))
    print("Total training time: %.3fs" % (time.time() - start_time))
    return costs

costs = train_batch(500)
predict = t.function([input_var], prediction)
px = predict(X)

def show_predict(i):
    plt.imshow(px[i,:].reshape(imgshape))
    plt.show()