#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os.path
import shutil
import time

import matplotlib.pylab as plt

from meteography import analysis
from meteography.neighbors import NearestNeighbors
from meteography.dataset import ImageSet, DataSet

base_dir = os.path.dirname(os.path.dirname(__file__))
hdf5_path = os.path.join(base_dir, 'temp', 'imgset.h5')
result_path = os.path.join(base_dir, 'temp', 'results')

parser = argparse.ArgumentParser()
parser.add_argument('datapath')
parser.add_argument('--hdf5', dest='hdf5path', default=hdf5_path)
parser.add_argument('--results', dest='resultpath', default=result_path)
args = parser.parse_args()

filenames = os.listdir(args.datapath)
filenames.sort()

imageset = ImageSet.create(args.hdf5path, (80, 117, 3))
dataset = DataSet.create(imageset.fileh, imageset)
onlineset = dataset.make_set('online')
neighbors = NearestNeighbors()
neighbors.fit(onlineset.input, onlineset.output)

if os.path.exists(args.resultpath):
    shutil.rmtree(args.resultpath)
os.mkdir(args.resultpath)

pred_times = []
expects = [None] * len(onlineset._v_attrs.intervals)
e = 0
for filename in filenames:
    filepath = os.path.join(args.datapath, filename)
    img = dataset.add_image(onlineset, filepath)
    expected = expects[e]
    expects[e] = imageset.recover_images([img['data']])[0]
    e = (e + 1) % len(expects)

    new_input = dataset.make_input(onlineset, img)
    if new_input is not None and len(onlineset.input) > 0:
        start_time = time.time()
        output = neighbors.predict(new_input)
        pred_times.append(time.time() - start_time)
        p = len(pred_times)
        print("Pred° %d in %.3fs" % (p, pred_times[-1]))
        output = imageset.recover_images([output])[0]
        imgcmp = analysis.compare_outputs(expected, output, dataset.img_shape)
        cmpfile = os.path.join(args.resultpath, "%d.jpg" % p)
        plt.imsave(cmpfile, imgcmp)
        if p % 300 == 0:
            start_time = time.time()
            dataset.reduce_dim()
            print("Dim reduction in %.3fs" % (time.time() - start_time))
            neighbors.fit(onlineset.input, onlineset.output)
imageset.close()
