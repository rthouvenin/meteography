# -*- coding: utf-8 -*-
from contextlib import contextmanager
from datetime import datetime
import logging
from numbers import Number
import os.path
import shutil
import threading

from django.core.files.storage import FileSystemStorage
import matplotlib.pylab as plt
from PIL import Image

from meteography.dataset import RawFeatures, PCAFeatures, RBMFeatures
from meteography.dataset import ImageSet, DataSet
from meteography.django.broadcaster import settings
from meteography.django.broadcaster.request_cache import get_dataset_cache

logger = logging.getLogger(__name__)


# Registry to instantiate features extractors from a name
features_extractors = {
    'raw': RawFeatures,
    'pca': PCAFeatures,
    'rbm': RBMFeatures,
}


# FIXME turn into actual storage
class WebcamStorage:
    PICTURE_DIR = 'pics'
    PREDICTION_DIR = 'predictions'

    def __init__(self, location):
        self.fs = FileSystemStorage(location)

    def dataset_path(self, webcam_id):
        return self.fs.path(webcam_id + '.h5')

    def _image_path(self, webcam_id, img_dir, timestamp=None):
        rel_path = os.path.join(webcam_id, img_dir)
        if timestamp is not None:
            path_format = settings.PICTURE_PATH.replace('%t', str(timestamp))
            pic_date = datetime.fromtimestamp(timestamp)
            pic_path = pic_date.strftime(path_format)
            rel_path = os.path.join(rel_path, pic_path)
        return rel_path

    def picture_path(self, webcam_id, timestamp=None):
        """
        Return the path of a picture from the given webcam
        and taken at given `timestamp`.
        If `timestamp` is None, return the  of the directory.
        """
        return self._image_path(webcam_id, self.PICTURE_DIR, timestamp)

    def prediction_path(self, webcam_id, params_name, timestamp=None):
        """
        Return the path of a prediction image for the given webcam, the
        prediction parameters named `params_name` and made at `timestamp`.
        If `timestamp` is None, return the path of the directory.
        """
        img_dir = os.path.join(self.PREDICTION_DIR, params_name)
        return self._image_path(webcam_id, img_dir, timestamp)

    def get_pixels(self, img, webcam_id):
        """
        Return the pixels data of an image, as given by
        `ImageSet.pixels_from_file()`.

        Parameters
        ----------
        img : int or str
            If a number, it is assumed to be a timestamp of when the picture
            was taken by the webcam. It will be read from the images set.
            If a string, it is assumed to be a path, it will be read from the
            file at this path.
        webcam_id : str
            The id of the webcam that took the picture
        """
        with self.get_dataset(webcam_id) as dataset:
            if isinstance(img, Number):
                img_dict = dataset.imgset.get_image(img, False)
                return img_dict['pixels']
            else:
                if not os.path.isabs(img):
                    img = self.fs.path(img)
                return dataset.imgset.pixels_from_file(img)

    def add_webcam(self, webcam_id):
        """
        Create the required files and directories for a new webcam
        """
        logger.info("Creating webcam %s on file system", webcam_id)

        # create pytables files
        hdf5_path = self.dataset_path(webcam_id)
        w, h = settings.WEBCAM_SIZE
        img_shape = h, w, 3
        w, h = settings.DEFAULT_FEATURES_SIZE
        feat_shape = h, w, 3
        with ImageSet.create(hdf5_path, img_shape) as imageset:
            extractor = RawFeatures(feat_shape, img_shape)
            imageset.add_feature_set(extractor)
            DataSet.create(imageset).close()

        # create directories for pictures
        pics_path = self.fs.path(self.picture_path(webcam_id))
        os.makedirs(pics_path)

    def delete_webcam(self, webcam_id):
        """
        Delete the files and directories related to a webcam
        """
        logger.info("Deleting webcam %s from file system", webcam_id)

        hdf5_path = self.dataset_path(webcam_id)
        os.remove(hdf5_path)
        shutil.rmtree(self.fs.path(webcam_id))

    def add_feature_set(self, feature_set_model):
        feat_type = feature_set_model.extract_type
        webcam_id = feature_set_model.webcam.webcam_id

        if feat_type not in features_extractors:
            raise ValueError("No features extractor named %s" % feat_type)

        def task(webcam_id, feat_type):
            with self.open_dataset(webcam_id) as dataset:
                if feat_type == 'raw':
                    img_shape = dataset.imgset.img_shape
                    w, h = settings.DEFAULT_FEATURES_SIZE
                    feat_shape = h, w, 3
                    extractor = RawFeatures(feat_shape, img_shape)
                elif feat_type == 'pca':
                    logger.info("Starting computation of a PCM model")
                    sample = dataset.imgset.sample()
                    extractor = PCAFeatures.create(sample)
                else:  # rbm
                    logger.info("Starting computation of a RBM model")
                    sample = dataset.imgset.sample()
                    extractor = RBMFeatures.create(sample)

                if extractor.name in dataset.imgset.feature_sets:
                    raise ValueError("The feature set %s already exists" %
                                     extractor.name)

                logger.info("Adding a new set of features %s to webcam %s",
                            feat_type, webcam_id)
                dataset.imgset.add_feature_set(extractor)

        t = threading.Thread(target=task, args=[webcam_id, feat_type])
        t.start()
        return feat_type  # FIXME an event should send name to model

    def delete_feature_set(self, feature_set_model):
        logger.info("Deleting feature set %s", feature_set_model.name)
        webcam_id = feature_set_model.webcam.webcam_id
        with self.get_dataset(webcam_id) as dataset:
            dataset.imgset.remove_feature_set(feature_set_model.name)
            dataset.repack()

    def open_dataset(self, webcam_id):
        hdf5_path = self.dataset_path(webcam_id)
        return DataSet.open(hdf5_path)

    @contextmanager
    def get_dataset(self, webcam_id):
        """
        Returns the dataset of `webcam_id`, cached for the current request.
        This method should be called only in a request thread.
        Returns a dummy context manager for backward compatibility
        """
        cache = get_dataset_cache()
        yield cache[self.dataset_path(webcam_id)]

    def add_picture(self, webcam, timestamp, fp):
        """
        Add a new picture associated to `webcam`

        Parameters
        ----------
        webcam : models.Webcam instance
        timestamp : str or int
            The UNIX Epoch of when the picture was taken
        fp : str or `file-like` object
            The filename or pointer to the file of the image.
            If a file object, it must be accepted by Pillow
        """
        # read and resize the image
        img = Image.open(fp)
        if img.size != settings.WEBCAM_SIZE:
            img = img.resize(settings.WEBCAM_SIZE)

        # store the image in file
        filepath = self.picture_path(webcam.webcam_id, timestamp)
        dirname = self.fs.path(os.path.dirname(filepath))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with self.fs.open(filepath, mode='wb') as fp_res:
            img.save(fp_res)

        # store the image in dataset
        with self.get_dataset(webcam.webcam_id) as dataset:
            # FIXME give directly PIL reference
            abspath = self.fs.path(filepath)
            img_dict = dataset.add_image(abspath)
            return img_dict

    def add_prediction(self, prediction):
        dirname = os.path.dirname(self.fs.path(prediction.path))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with self.fs.open(prediction.path, 'w') as fp:
            plt.imsave(fp, prediction.sci_bytes)

    def add_examples_set(self, params):
        """
        Create the directories and pytables group for a set of examples
        """
        def task(params):
            logger.info("Creating example set %s on file system", params.name)
            cam_id = params.webcam.webcam_id
            pred_path = self.fs.path(self.prediction_path(cam_id, params.name))
            try:
                # Make sure the directory is empty if it exists
                shutil.rmtree(pred_path, ignore_errors=True)
                os.makedirs(pred_path)

                with self.open_dataset(cam_id) as dataset:
                    dataset.make_set(params.name, params.intervals,
                                     params.features.name)
            except Exception:
                logger.exception("Error while creating set %s", params.name)
            else:
                logger.info("Done creating the set %s", params.name)

        t = threading.Thread(target=task, args=[params])
        t.start()

    def delete_examples_set(self, params):
        """
        Remove the directories and pytables group for a set of examples
        """
        logger.info("Deleting example set %s from file system", params.name)

        cam_id = params.webcam.webcam_id
        pred_path = self.fs.path(self.prediction_path(cam_id, params.name))
        try:
            shutil.rmtree(pred_path)
            with self.get_dataset(cam_id) as dataset:
                dataset.delete_set(params.name)
                dataset.repack()
        except Exception:
            logger.exception("Error while deleting set %s", params.name)
        else:
            logger.info("Done deleting the set %s", params.name)

# Default storage instance
webcam_fs = WebcamStorage(settings.WEBCAM_ROOT)
