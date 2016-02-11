# -*- coding: utf-8 -*-

from datetime import datetime

from sklearn.neighbors import DistanceMetric

from django.utils.timezone import utc

from meteography.neighbors import NearestNeighbors
from meteography.django.broadcaster.models import Prediction
from meteography.django.broadcaster.storage import webcam_fs


def make_prediction(webcam, params, timestamp):
    """
    Make a prediction using NearestNeighbors algorithm.

    Parameters
    ----------
    webcam : Webcam instance
        The source of pictures
    params : PredictionParams instance
        The parameters to use to compute the prediction
    timestamp : int
        The Unix-Epoch-timestamp (UTC) of when the prediction is made

    Return
    ------
    Prediction instance
    """
    cam_id = webcam.webcam_id
    result = None
    with webcam_fs.get_dataset(cam_id) as dataset:
        ex_set = dataset.get_set(params.name)
        new_input = dataset.make_input(ex_set, timestamp)
        if new_input is not None and len(ex_set.input) > 0:
            neighbors = NearestNeighbors()
            neighbors.fit(ex_set.input)
            output_ref = neighbors.predict(new_input)
            output = dataset.output_img(ex_set, output_ref)
            # FIXME reference existing image (data and file)
            imgpath = webcam_fs.prediction_path(cam_id, params.name, timestamp)
            result = Prediction(params=params, path=imgpath)
            result.comp_date = datetime.fromtimestamp(timestamp, utc)
            result.sci_bytes = output
            result.create()
    return result


def update_prediction(prediction, real_pic, metric_name='euclidean'):
    """
    Update a prediction after receiving the actual picture from the webcam.

    Parameters
    ----------
    prediction : Prediction
        The model object of the prediction to update
    real_pic : Picture
        The model object of the actual picture received

    Return
    ------
    float : the prediction error
    """
    pred_pic = prediction.as_picture()
    cam_id = prediction.params.webcam.webcam_id
    if metric_name == 'wminkowski-pca':
        with webcam_fs.get_dataset(cam_id) as dataset:
            if 'pca' not in dataset.imgset.feature_sets:
                raise ValueError("""wminkowski-pca cannnot be used
                                    without a PCA feature set""")

            pca_extractor = dataset.imgset.feature_sets['pca'].extractor
            weights = pca_extractor.pca.explained_variance_ratio_
            pred_data = pca_extractor.extract(pred_pic.pixels)
            real_data = pca_extractor.extract(real_pic.pixels)
            metric = DistanceMetric.get_metric('wminkowski', p=2, w=weights)
    else:
        pred_data = pred_pic.pixels
        real_data = real_pic.pixels
        metric = DistanceMetric.get_metric(metric_name)

    error = metric.pairwise([pred_data], [real_data])[0]
    prediction.error = error
    prediction.save()
    return error
