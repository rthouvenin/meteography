#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from meteography import analysis
from meteography import dataset
from meteography.dataset import DataSet
from meteography.dataset import ImageSet

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

datapath = os.path.join('..', 'data', 'webcams', 'tinycam')
resultpath = 'results'

logging.info("Creating the image set...")
start_time = time.time()
fp = dataset.create_imagegroup('/home/romain/tmp/temp.h5', (80, 117, 3))
imageset = ImageSet(fp)
imageset.add_images(datapath)
logging.info("Done in %.4fs" % (time.time() - start_time))
start_time = time.time()
logging.info("Reducing the image set...")
imageset.reduce_dim()
logging.info("Done in %.4fs" % (time.time() - start_time))
start_time = time.time()
logging.info("Creating the data set...")
dataset = DataSet.make(imageset)
dataset.split(.8, .2)
logging.info("Done in %.4fs" % (time.time() - start_time))
#saving it for later
expectation = imageset.recover_images(dataset.valid_output)
logging.info("Fitting the training set...")
algo = NearestNeighbors(1, algorithm='brute')
algo.fit(dataset.train_input)

logging.info("Making predictions for the validation set...")
start_time = time.time()
prediction_ids = algo.kneighbors(dataset.valid_input, return_distance=False)
query_time = (time.time() - start_time) / len(dataset.valid_input)
prediction_ids = prediction_ids.reshape((len(prediction_ids), ))
prediction = imageset.recover_images(dataset.train_output[prediction_ids])
error = np.linalg.norm(expectation - prediction).mean()
logging.info("Average error: %.3f, average query time: %.4fs"
             % (error, query_time))

logging.info("Storing results...")
os.mkdir(resultpath)
test_ids = range(len(prediction))
for i, expected, predicted in zip(test_ids, expectation, prediction):
    filename = os.path.join(resultpath, "%d.jpg" % i)
    imgcmp = analysis.compare_outputs(expected, predicted, dataset.img_shape)
    plt.imsave(filename, imgcmp)

fp.close()
