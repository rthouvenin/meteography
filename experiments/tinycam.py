# -*- coding: utf-8 -*-
"""
Script to experiment with the dataset tinycam, which is made of real images
reduced to a small size (but bigger than rednight and bigrednight)
There are also more examples than in previous experiments.
The aim is to play with a bigger dataset with more realistic images,
and also try other types of learning.
"""

import logging
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from meteography import analysis
from meteography.dataset import DataSet
from meteography.dataset import ImageSet
from meteography.perceptron import Perceptron

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

datapath = os.path.join('..', 'data', 'webcams', 'tinycam')

imageset = ImageSet.create('/home/romain/tmp/temp.h5', (80, 117, 3))
imageset.add_images(datapath)
dataset = DataSet.create(imageset.fileh, imageset)
dataset.make(interval=1800)
imageset.close()
insize = dataset.input_data.shape[1]
outsize = dataset.output_data.shape[1]
mlp = Perceptron(insize, outsize)

dataset.split(.8, .2)
analysis.plot_learning_curve(dataset, mlp, 300)

costs, elapsed = mlp.train(dataset.input_data, dataset.output_data)
plt.plot(costs)

y = dataset.output_data
p = mlp.predict(dataset.input_data)
vline = np.zeros((80, 5))
for i in range(0, len(y), 10):
    yi = y[i].reshape((80, 117))
    pi = p[i].reshape((80, 117))
    cmpimg = np.hstack([yi, vline, pi])
    plt.imsave('results/%04d.jpg' % i, cmpimg, cmap=matplotlib.cm.Greys_r)
