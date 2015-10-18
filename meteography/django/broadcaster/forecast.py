# -*- coding: utf-8 -*-

import matplotlib.pylab as plt

from meteography.neighbors import NearestNeighbors
from meteography.django.broadcaster.storage import webcam_fs


def make_prediction(webcam, params, timestamp):
    cam_id = webcam.webcam_id
    with webcam_fs.get_dataset(cam_id) as dataset:
        onlineset = dataset.get_set(params.name)
        new_input = dataset.make_input(onlineset, int(timestamp))
        if new_input is not None and len(onlineset.input) > 0:
            neighbors = NearestNeighbors()
            neighbors.fit(onlineset.input, onlineset.output)
            output = neighbors.predict(new_input).reshape(dataset.img_shape)
            plt.imsave(webcam_fs.prediction_path(cam_id, timestamp), output)
