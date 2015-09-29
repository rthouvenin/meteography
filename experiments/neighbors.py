# -*- coding: utf-8 -*-

import logging
import os
import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from meteography import analysis
from meteography.dataset import DataSet
from meteography.dataset import ImageSet

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)

datapath = os.path.join('..', 'data', 'webcams', 'tinycam')
resultpath = 'results'

print("Creating the dataset...")
start_time = time.time()
imageset = ImageSet.make(datapath)
imageset.reduce_dim()
dataset = DataSet.make(imageset)
dataset.split(.8, .2)
#saving it for later
expectation = imageset.recover_images(dataset.valid_output)
print("Done in %.4fs" % (time.time() - start_time))
print("Fitting the training set...")
algo = NearestNeighbors(1, algorithm='brute')
algo.fit(dataset.train_input)

print("Making predictions for the validation set...")
start_time = time.time()
prediction_ids = algo.kneighbors(dataset.valid_input, return_distance=False)
query_time = (time.time() - start_time) / len(dataset.valid_input)
prediction_ids = prediction_ids.reshape((len(prediction_ids), ))
prediction = imageset.recover_images(dataset.train_output[prediction_ids])
error = np.linalg.norm(expectation - prediction).mean()
print("Average error: %.3f, average query time: %.4fs" % (error, query_time))

print("Storing results...")
os.mkdir(resultpath)
test_ids = range(len(prediction))
for i, expected, predicted in zip(test_ids, expectation, prediction):
    filename = os.path.join(resultpath, "%d.jpg" % i)
    imgcmp = analysis.compare_outputs(expected, predicted, dataset.img_shape)
    plt.imsave(filename, imgcmp)
