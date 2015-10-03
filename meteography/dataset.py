# -*- coding: utf-8 -*-
"""
Classes and associated helper functions to create and manipulate the data sets
used in the machine learning machinery
"""

import bisect
from datetime import datetime
import logging
import os.path
import pickle
import random
import time

import numpy as np
import PIL
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import tables
from tables.nodes import filenode

MAX_PIXEL_VALUE = 255
COLOR_BANDS = ('R', 'G', 'B')
GREY_BANDS = ('L', )
PIXEL_TYPE = np.float16

logger = logging.getLogger(__name__)


def image_descriptor(img_size):
    """
    Create a pytables descriptor for the table of images.

    Parameters
    ----------
    img_size : int
        The number of pixels multiplied by the number of bands
    """
    return {
        'name': tables.StringCol(256),
        'time': tables.UIntCol(),
        'pixels': tables.Col.from_sctype(PIXEL_TYPE, img_size)
    }


def parse_timestamp(filename):
    """
    File name parser when the name without extension is a Unix Epoch
    """
    basename = os.path.basename(filename)
    strtime = basename[:basename.index('.')]
    return int(strtime)


def parse_path(filename):
    """
    File name parser when the time data is in the path to the file like so:
    <basedir>/<year>/<month>/<day>/<hour>-<minute>.<ext>
    """
    elems = filename.split(os.path.sep)
    year = int(elems[-4])
    month = int(elems[-3])
    day = int(elems[-2])
    basename = elems[-1]
    hour, minute = basename[:basename.index('.')].split('-')
    date = datetime(year, month, day, int(hour), int(minute))
    return int(time.mktime(date.timetuple()))


class ImageSet:
    def __init__(self, fileh):
        """
        An ImageSet is a container for the complete list of webcam snapshots
        that can be used to train the learner. It is backed by a HDF5 file.

        Parameters
        ----------
        fileh : pytables file descriptor
            An opened file descriptor pointing to a HDF5 file. The file must
            already contain the required nodes, they can be created with the
            function `create_imagegroup`.

        Attributes
        ----------
        img_shape : tuple
            The shape of the images: (height, width[, bands])
        img_size : int
            The number of pixels in a single image. When the dimension is not
            reduced (see `reduce_dim`), this is the product of width, height
            and number of bands.
            When reduced, this is the reduced dimension.
        is_grey : bool
            True if the images are processed as greyscale images
        """
        self.fileh = fileh
        self.table = fileh.root.images.imgset
        self.img_shape = self.table.attrs.img_shape
        self.is_grey = len(self.img_shape) == 2
        if '/images/pcamodel' in self.fileh:
            fn = filenode.open_node(self.fileh.root.images.pcamodel, 'r')
            self.pca = pickle.load(fn)
            self.img_size = self.pca.components_.shape[0]
            fn.close()
        else:
            self.pca = None
            self.img_size = np.prod(self.img_shape)
        self._times = None
    
    @classmethod
    def create(cls, thefile, img_shape):
        """
        Create in `thefile` a pytables node 'images' to be used for image data.
        Create in the node a table 'imgset' of images of shape `img_shape`.
        Create a `ImageSet` backed by this file and return it.
    
        Parameters
        ----------
        thefile : str or pytables file descriptor
            The name of the file to create, or a file descriptor already
            opened. If the name of an existing file is given, it will be 
            overwritten.
        img_shape : tuple
            The shape of the images: (height, width[, bands])
        """
        if not hasattr(thefile, 'create_table'):
            fp = tables.open_file(thefile, mode='w')
        else:
            fp = thefile
        try:
            desc = image_descriptor(np.prod(img_shape))
            group = fp.create_group('/', 'images')
            table = fp.create_table(group, 'imgset', desc)
            table.attrs.img_shape = img_shape
            table.cols.time.create_csindex()
        except Exception as e:
            #Avoid an orphan open file in case of a problem
            if fp is not thefile:
                fp.close()
            raise e
        return cls(fp)
    
    @classmethod
    def open(cls, thefile):
        """
        Instantiate a ImageSet backed by `thefile`.
        
        Parameters
        ----------
        thefile : str or pytables file descriptor
            The name of the file to open, or a pytables file descriptor already
            opened. 
        """
        if not hasattr(thefile, 'create_table'):
            fp = tables.open_file(thefile, mode='a')
        else:
            fp = thefile
        return cls(fp)
    
    def close(self):
        return self.fileh.close()
    
    def _img_from_row(self, row, reduced=True):
        """
        Create from a row a dictionary with the details of the image in that
        row. If `reduced` is True and the ImageSet was reduced, the key 'data'
        will contain the PCA-transformed data.
        """
        if reduced and self.pca is not None:
            pca_pixels = self.fileh.root.images.pcapixels
            pixels = pca_pixels[row.nrow]
        else:
            pixels = row['pixels']
        return {
            'name': row['name'],
            'time': row['time'],
            'data': pixels
        }

    def get_image(self, t, reduced=True):
        """
        Returns as a dictionary the details of the image taken at time `t`.
        If `reduced` is True and the ImageSet was reduced, the key 'data'
        will contain the PCA-transformed data.
        """
        rows = self.table.where('time == ' + str(t))
        row = next(rows, None)
        if row:
            return self._img_from_row(row, reduced)
        return None

    def __iter__(self):
        """
        Create a generator on the images, returned as dictionaries.
        """
        rows = self.table.iterrows()
        for row in rows:
            yield self._img_from_row(row)

    def __len__(self):
        """Return the number of images in the set."""
        return len(self.table.nrows)

    def _add_image(self, imgfile, name_parser):
        """
        `add_image` without flushing the table.
        """
        try:
            with open(imgfile, 'rb') as fp:
                img = PIL.Image.open(fp)
                img_shape = img.size[1], img.size[0]
                if img_shape != self.img_shape[:2]:
                    raise ValueError("The image has shape %s instead of %s"
                                     % (img_shape, self.img_shape[:2]))
                #Convert the image to greyscale or RGB if necessary
                expected_bands = GREY_BANDS if self.is_grey else COLOR_BANDS
                img_bands = img.getbands()
                if img_bands != expected_bands:
                    img = img.convert(''.join(expected_bands))
                data = np.array(img, dtype=PIXEL_TYPE) / MAX_PIXEL_VALUE
            #Flatten the data in case of RGB tuples
            if data.ndim > 1:
                data = data.flatten()
            img_time = name_parser(imgfile)
            row = self.table.row
            row['name'] = imgfile
            row['time'] = img_time
            row['pixels'] = data
            row.append()
            self._times = None  # invalidate times cache
        except Exception as e:
            logger.warn("Ignoring file %s: %s" % (imgfile, e))

    def add_image(self, imgfile, name_parser=parse_timestamp):
        """
        Add an image to the set.

        Parameters
        ----------
        imgfile : str
            The file name of the image
        name_parser : callable
            A function that takes an absolute filename as argument and
            returns the timestamp of when the photo was taken
        """
        self._add_image(imgfile, name_parser)
        self.table.flush()

    def add_images(self, directory, name_parser=parse_timestamp,
                   recursive=True):
        """
        Add the images of a directory to the set

        Parameters
        ----------
        directory : str
            The path of the directory
        name_parser : callable
            A function that takes an absolute filename as argument and
            returns the timestamp of when the photo was taken
        recursive : bool
            Whether to scan the sub-directories recursively
        """
        filenames = os.listdir(directory)
        for f in filenames:
            fullname = os.path.join(directory, f)
            if os.path.isdir(fullname):
                if recursive:
                    self.add_images(fullname, name_parser)
            else:
                self._add_image(fullname, name_parser)
        self.table.flush()

#    def sort(self): Requires pcapixels sorting as well
#        newtable = self.table.copy(newname='sortedimgset', sortby='time')
#        self.table.remove()
#        newtable.rename('imgset')
#        self.table = newtable
#        self.fileh.flush()
#        self.sorted = True

    def _sample(self, sample_size):
        """
        Pick randomly `sample_size` images and return their pixels as a single
        ndarray of shape (`sample_size`, `img_size`).
        """
        if sample_size == self.table.nrows:
            sample = self.table.cols.pixels[:]
        else:
            index = random.sample(xrange(self.table.nrows), sample_size)
            sample = np.empty((sample_size, self.img_size), dtype=PIXEL_TYPE)
            for i, img in enumerate(self.table.itersequence(index)):
                sample[i] = img['pixels']
        return sample

    def reduce_dim(self, sample_size=1000):
        """
        Apply PCA transformation to each image of the set.

        Parameters
        ----------
        sample_size : number or None
            Indicates how many images should be used to compute the PCA
            reduction. If 0 or None, all the images of the set are used. If it
            is a float in ]0, 1], it indicates the proportion of images to be
            used. If greater than 1, it indicates the maximum number of images
            to use.
        """
        nb_images = self.table.nrows
        if not sample_size:
            sample_size = nb_images
        elif 0 < sample_size <= 1:
            sample_size = int(nb_images * sample_size)
        else:
            sample_size = min(nb_images, sample_size)

        #Compute PCA components and save them
        sample = self._sample(sample_size)
        self.pca = PCA().fit(sample) # FIXME: choose an algo
        ##self.pca = TruncatedSVD(min(300, sample_size)).fit(sample)
        fn = filenode.new_node(self.fileh, where='/images', name='pcamodel')
        pickle.dump(self.pca, fn, pickle.HIGHEST_PROTOCOL)
        fn.close()

        #Apply PCA transformation to all the images in chunks
        self.img_size = self.pca.components_.shape[0]
        pixels = self.table.cols.pixels
        pca_pixels = self.fileh.create_array('/images', 'pcapixels',
                                             shape=(nb_images, self.img_size),
                                             atom=tables.FloatAtom())
        chunk_size = sample_size
        nb_chunks = nb_images / chunk_size
        for c in range(nb_chunks):
            s = c * chunk_size
            e = min(nb_images, s + chunk_size)
            pca_pixels[s:e] = self.pca.transform(pixels[s:e])
            pca_pixels.flush()

    def recover_images(self, pca_pixels):
        """
        Appy inverse PCA transformation to the given transformed data.

        Parameters
        ----------
        pca_pixels : array
        """
        pixels_out = self.pca.inverse_transform(pca_pixels)
        return pixels_out.clip(0, 1, pixels_out)

    def find_closest(self, start, interval):
        """
        Search for the image with its time attribute the closest to
        `start+interval`, constrained in `]start, start+2*interval]`.
        This is a binary search in O(log n), except the first time it is called

        Return
        ------
        The image (a `dict`) or `None` if none is found
        """
        assert start >= 0 and interval > 0
        target = start + interval
        maxtarget = target + interval
        if self._times is None:
            self._times = self.table.cols.time[:]
            self._times.sort()

        idx = bisect.bisect(self._times, start+interval)
        left_time = self._times[idx-1]  # left_time <= target < maxtarget
        if idx == len(self._times):
            right_time = maxtarget+1  # unreachable
        else:
            right_time = self._times[idx]  # right_time > target

        if target - left_time < right_time - target:
            if left_time > start:
                return self.get_image(left_time)
            else:  # At this point we can deduce that right_time > maxtarget
                return None
        else:
            if right_time <= maxtarget:
                return self.get_image(right_time)
            else:  # At this point we can deduce that left_time < start
                return None


class DataSet:
    def __init__(self, imageset, indata, outdata):
        """
        Build a DataSet from the images of `imageset`.
        The input and target data are built, but not split into training,
        validation and test sets (use `split` for that).

        Parameters
        ----------
        imageset : ImageSet
            The ImageSet containing all the source images.
        """
        self.imgset = imageset
        self.input_data = indata
        self.output_data = outdata
        self.img_shape = imageset.img_shape
        self.img_size = imageset.img_size
        self.history_len = indata.shape[1] % self.img_size
        self.is_split = False

    @classmethod
    def make(cls, images, hist_len=3, interval=600, future_time=1800):
        """
        Constructs a DataSet from the ImageSet `images`.

        Parameters
        ----------
        imageset : ImageSet
            The ImageSet containing all the source images.
        hist_len : int
            The number of images to include in the input data
        interval : int
            The number of seconds between each input image
        future_time : int
            The number of seconds between the latest input image
            and the target image
        """
        #Find the representations of the data points
        data_points = DataSet._find_datapoints(images, hist_len, interval,
                                               future_time)
        #Copy the data into numpy arrays
        imgdim = images.img_size
        nb_ex = len(data_points)
        nb_feat = (imgdim+1) * hist_len
        input_data = np.ndarray((nb_ex, nb_feat), dtype=PIXEL_TYPE)
        output_data = np.ndarray((nb_ex, imgdim), dtype=PIXEL_TYPE)
        for i, data_point in enumerate(data_points):
            example, target = data_point
            for j, img in enumerate(example):
                input_data[i, imgdim*j:imgdim*(j+1)] = img['data']
                input_data[i, j-hist_len] = target['time'] - img['time']
            output_data[i] = target['data']
        return cls(images, input_data, output_data)

    @staticmethod
    def _find_datapoints(imgset, hist_len, interval, future_time):
        """
        Build a list of tuples (input_images, output_value) to be used for
        training or validation.

        Parameters
        ----------
        hist_len : int (strictly greater than 0)
            The number of input images in each data point
        interval : int (strictly greater than 0)
            The approximate (+/- 100%) time interval between each input image
        future_time : int (strictly greater than 0)
            The approximate (+/- 100%) time interval between the last input
            image and the output value

        Returns
        -------
        list(tuple(list(filedict), filedict)) : a list of tuples
            (input_images, output_value) where input_images is a list of images
            to be used as the input of an example, and output_value is the
            output (expected prediction) of the example
        """
        assert hist_len > 0 and interval > 0 and future_time > 0
        data_points = []
        for i, img in enumerate(imgset):
            images_i = [img]
            for j in range(1, hist_len):
                prev_time = images_i[j-1]['time']
                img_j = imgset.find_closest(prev_time, interval)
                if img_j:
                    images_i.append(img_j)
                else:
                    break
            if len(images_i) == hist_len:
                latest_time = images_i[-1]['time']
                target = imgset.find_closest(latest_time, future_time)
                if target:
                    data_points.append((images_i, target))
        return data_points

    def input_img(self, i, j):
        """
        Return the pixels the `j`th image from example `i` in the input data,
        with its original shape (directly displayable by matplotlib)
        """
        assert 0 <= i < len(self.input_data) and 0 <= j < self.history_len
        img = self.input_data[i, j*self.img_size:(j+1)*self.img_size]
        return img.reshape(self.img_shape)

    def output_img(self, i):
        """
        Return the pixels the `i`th output image,
        with its original shape (directly displayable by matplotlib)
        """
        assert 0 <= i < len(self.output_data)
        img = self.output_data[i]
        return img.reshape(self.img_shape)

    def split(self, train=.7, valid=.15):
        """
        Split the examples into a training set, validation set and test set.
        The data is shuffled before the split, and the original order is lost.
        The method can be called multiple times to apply multiple shuffles and
        obtain different training/validation/test splits.
        The examples allocated to the test set are the remaining ones after
        creating the training and validation sets

        Parameters
        ----------
        train : float in [0, 1]
            The proportion of examples to allocate to the training set
        valid : float in [0, 1], with `train` + `valid` <= 1
            The proportion of examples to allocate to the validation set
        """
        assert 0 <= (train + valid) <= 1
        full_size = len(self.input_data)
        indexes = range(full_size)
        np.random.shuffle(indexes)
        self.input_data = self.input_data[indexes]
        self.output_data = self.output_data[indexes]
        train_size = int(full_size * train)
        valid_size = int(full_size * valid)
        self.__dispatch_data(train_size, valid_size)
        self.is_split = train_size, valid_size

    def __dispatch_data(self, train_size, valid_size, dispatch_output=True):
        #These are views, not copies
        endvalid = train_size + valid_size
        self.train_input = self.input_data[:train_size]
        self.valid_input = self.input_data[train_size:endvalid]
        self.test_input = self.input_data[train_size+valid_size:]
        if dispatch_output:
            self.train_output = self.output_data[:train_size]
            self.valid_output = self.output_data[train_size:endvalid]
            self.test_output = self.output_data[train_size+valid_size:]

    def training_set(self):
        "Return the tuple input, output of the training set"
        if self.is_split:
            return self.train_input, self.train_output
        return self.input_data, self.output_data

    def validation_set(self):
        "Return the tuple input, output of the validation set"
        if self.is_split:
            return self.valid_input, self.valid_output
        return [], []

    def test_set(self):
        "Return the tuple input, output of the test set"
        if self.is_split:
            return self.test_input, self.test_output
        return [], []

    def save(self, path):
        """
        Save this DataSet in a file.

        Parameters
        ----------
        path : str
            the file name. If the file does not exist it is created
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def __getstate__(self):
        odict = self.__dict__.copy()
        split_keys = ['train_input', 'valid_input', 'test_input',
                      'train_output', 'valid_output', 'test_output']
        odict.update(dict.fromkeys(split_keys))
        return odict

    @staticmethod
    def load(path):
        """
        Create a DataSet from a file created with the `save` method

        Parameters
        ----------
        path : str
            the file name
        """
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
            #Re-create training, validation and test views on the data
            if dataset.is_split:
                dataset.__dispatch_data(*dataset.is_split)
            return dataset
