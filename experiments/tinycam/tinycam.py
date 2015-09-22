# -*- coding: utf-8 -*-
"""
Script to experiment with the dataset tinycam, which is made of real images
reduced to a small size (but bigger than rednight and bigrednight)
There are also more examples than in previous experiments.
The aim is to play with a bigger dataset, and also try instance-based learning.
"""

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from meteography.dataset import DataSet
from meteography.perceptron import Perceptron

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

dataset = DataSet.make('data', interval=30)
insize = dataset.input_data.shape[1]
outsize = dataset.output_data.shape[1]
mlp = Perceptron(insize, outsize)
costs, elapsed = mlp.train(dataset.input_data, dataset.output_data)

y = dataset.output_data
p = mlp.predict(dataset.input_data)
vline = np.zeros((80, 5))
for i in range(0, len(y), 10):
     yi = y[i].reshape((80,117))
     pi = p[i].reshape((80,117))
     cmp = np.hstack([yi, vline, pi])
     plt.imsave(fname='results/%4d.jpg' % i, arr=cmp, cmap=matplotlib.cm.Greys_r)