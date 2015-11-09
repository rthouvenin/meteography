# -*- coding: utf-8 -*-

from datetime import datetime

from django.utils.timezone import utc

from meteography.neighbors import NearestNeighbors
from meteography.django.broadcaster.models import Prediction
from meteography.django.broadcaster.storage import webcam_fs


def make_prediction(webcam, params, timestamp):
    cam_id = webcam.webcam_id
    result = None
    with webcam_fs.get_dataset(cam_id) as dataset:
        onlineset = dataset.get_set(params.name)
        new_input = dataset.make_input(onlineset, int(timestamp))
        if new_input is not None and len(onlineset.input) > 0:
            neighbors = NearestNeighbors()
            neighbors.fit(onlineset.input)
            output_ref = neighbors.predict(new_input)
            output = dataset.output_img(onlineset, output_ref)
            # FIXME reference existing image (data and file)
            imgpath = webcam_fs.prediction_path(cam_id, params.name, timestamp)
            result = Prediction(params=params)
            result.comp_date = datetime.fromtimestamp(float(timestamp), utc)
            result.sci_bytes = output
            result.path = imgpath
    return result
