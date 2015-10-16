# -*- coding: utf-8 -*-
import os.path

from django.core.files.storage import FileSystemStorage
from PIL import Image

from meteography.dataset import ImageSet, DataSet
from meteography.django.broadcaster import settings


class WebcamStorage:
    PICTURE_DIR = 'pics'

    def __init__(self, location=settings.WEBCAM_DIR):
        self.fs = FileSystemStorage(location)

    def dataset_path(self, webcam_id):
        return webcam_id + '.h5'

    def picture_path(self, webcam_id, timestamp=None):
        """
        Return the path (relative to the location of this storage) of
        a picture from the given webcam and taken at given `timestamp`.
        If `timestamp` is None, return the path directory.
        """
        if timestamp is None:
            return os.path.join(webcam_id, self.PICTURE_DIR)
        else:
            basename = '%s.jpg' % str(timestamp)
            return os.path.join(webcam_id, self.PICTURE_DIR, basename)

    def add_picture(self, webcam_id, timestamp, fp):
        """
        Add a new picture associated to webcam `webcam_id`

        Parameters
        ----------
        webcam_id : str
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
        filepath = self.picture_path(webcam_id, timestamp)
        with self.fs.open(filepath, mode='wb') as fp_res:
            img.save(fp_res)

        # store the image in dataset
        hdf5_path = self.fs.path(self.dataset_path(webcam_id))
        with DataSet.open(hdf5_path) as dataset:
            # FIXME give directly PIL reference
            dataset.add_image(settings.SET_NAME, self.fs.path(filepath))

    def add_webcam(self, webcam_id):
        """
        Create the required files and directories for a new webcam
        """
        hdf5_path = self.fs.path(self.dataset_path(webcam_id))
        w, h = settings.WEBCAM_SIZE
        img_shape = h, w, 3
        with ImageSet.create(hdf5_path, img_shape) as imageset:
            with DataSet.create(imageset) as dataset:
                dataset.make_set(settings.SET_NAME)  # FIXME Configurabilize
        pics_path = self.fs.path(self.picture_path(webcam_id))
        os.makedirs(pics_path)
