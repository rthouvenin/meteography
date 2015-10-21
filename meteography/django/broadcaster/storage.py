# -*- coding: utf-8 -*-
import os.path

from django.core.files.storage import FileSystemStorage
from PIL import Image

from meteography.dataset import ImageSet, DataSet
from meteography.django.broadcaster import settings


# FIXME turn into actual storage
class WebcamStorage:
    PICTURE_DIR = 'pics'
    PREDICTION_DIR = 'predictions'

    def __init__(self, location):
        self.fs = FileSystemStorage(location)

    def dataset_path(self, webcam_id):
        return self.fs.path(webcam_id + '.h5')

    def _image_path(self, webcam_id, img_dir, timestamp=None):
        if timestamp is None:
            rel_path = os.path.join(webcam_id, img_dir)
        else:
            basename = '%s.jpg' % str(timestamp)
            rel_path = os.path.join(webcam_id, img_dir, basename)
        #return self.fs.path(rel_path)
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

    def add_webcam(self, webcam_id):
        """
        Create the required files and directories for a new webcam
        """
        # create pytables files
        hdf5_path = self.dataset_path(webcam_id)
        w, h = settings.WEBCAM_SIZE
        img_shape = h, w, 3
        with ImageSet.create(hdf5_path, img_shape) as imageset:
            DataSet.create(imageset).close()

        # create directories for pictures
        pics_path = self.fs.path(self.picture_path(webcam_id))
        os.makedirs(pics_path)

    def get_dataset(self, webcam_id):
        hdf5_path = self.dataset_path(webcam_id)
        return DataSet.open(hdf5_path)

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
        with self.fs.open(filepath, mode='wb') as fp_res:
            img.save(fp_res)

        # store the image in dataset
        with self.get_dataset(webcam.webcam_id) as dataset:
            # FIXME give directly PIL reference
            dataset.add_image(self.fs.path(filepath))

    def add_examples_set(self, params):
        """
        Create the directories and pytables group for a set of examples
        """
        cam_id = params.webcam.webcam_id
        pred_path = self.fs.path(self.prediction_path(cam_id, params.name))
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        with self.get_dataset(cam_id) as dataset:
            dataset.init_set(params.name, intervals=params.intervals)

# Default storage instance
webcam_fs = WebcamStorage(settings.WEBCAM_ROOT)
