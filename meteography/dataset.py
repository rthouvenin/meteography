# -*- coding: utf-8 -*-
"""
Classes and associated helper functions to create and manipulate the data sets
used in the machine learning machinery
TODO: log ignored files
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

MAX_PIXEL_VALUE = 255
COLOR_BANDS = ('R', 'G', 'B')
GREY_BANDS = ('L', )
PIXEL_TYPE = np.float16

logger = logging.getLogger(__name__)


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


def create_filedict(filename, name_parser=parse_timestamp):
    """
    Open `filename` as an image and read some metadata.
    """
    filedict = {'name': filename}
    try:
        with open(filename, 'rb') as fp:
            img = PIL.Image.open(fp)
            filedict['shape'] = img.size[1], img.size[0]
            filedict['time'] = name_parser(filename)
    except Exception as e:
        filedict['error'] = e
    return filedict


def extract_image(file_dict, grayscale=False):
    """
    Read the pixels of the image in `filedict` and store them in a 'data'
    attribute as a flat array with the value of the pixels in [0,1]
    """
    expected_bands = GREY_BANDS if grayscale else COLOR_BANDS
    with open(file_dict['name'], 'rb') as fp:
        img = PIL.Image.open(fp)
        bands = img.getbands()
        if bands != expected_bands:
            img = img.convert(''.join(expected_bands))
        raw_data = img.getdata()
    data = np.asarray(raw_data, dtype=PIXEL_TYPE) / MAX_PIXEL_VALUE
    #Flatten the data in case of RGB tuples
    if len(data.shape) > 1:
        data = data.flatten()
    file_dict['data'] = data
    if not grayscale:
        file_dict['shape'] += (3, )
    return file_dict


class ImageSet:
    def __init__(self, images):
        self.images = sorted(images, key=lambda x: x['time'])
        self.times = [i['time'] for i in self.images]
        self.img_shape = images[0]['shape']
        self.img_size = np.prod(self.img_shape)

    @classmethod
    def make(cls, dirpath, greyscale=False, name_parser=parse_timestamp,
             recursive=True):
        """
        Build a ImageSet from the images located in the directory `dirpath`.
        All the files are attempted to be read, the ones that cannot be read as
        an image or whose time cannot be read are ignored.
        All the images are expected to have the same dimensions.

        Parameters
        ----------
        dirpath : str
            The source directory
        greyscale : bool
            Whether to load greyscale images (True) or color images (False)
        name_parser : callable
            A function that takes an absolute filename as argument and
            returns the timestamp of when the photo was taken
        recursive : bool
            Whether to scan the source directory recursively
        """
        files = ImageSet.create_filedicts(dirpath, name_parser, recursive)
        #Filter out junk
        imgshape = files[0]['shape']
        files = [f for f in files if f['shape'] == imgshape]
        return cls([extract_image(f, greyscale) for f in files])

    @staticmethod
    def create_filedicts(dirpath, name_parser, recursive=True, filedicts=[]):
        """
        Scans a directory and create a dictionary for each image found.
        The dictionaries are appended to the list `filedicts` (which is also
        returned)
        """
        filenames = os.listdir(dirpath)
        for f in filenames:
            fullname = os.path.join(dirpath, f)
            if os.path.isdir(fullname):
                if recursive:
                    ImageSet.create_filedicts(fullname, name_parser,
                                              True, filedicts)
            else:
                filedict = create_filedict(fullname, name_parser)
                if 'error' in filedict:
                    logger.warn("Ignoring file %s: %s"
                                % (fullname, filedict['error']))
                else:
                    filedicts.append(filedict)
        return filedicts

    def find_closest(self, start, interval):
        """
        Search for the image with its time attribute the closest to
        `start+interval`, constrained in `]start, start+2*interval]`.
        This is a binary search in O(log n)

        Return
        ------
        The image (a `dict`) or `None` if none is found
        """
        assert start >= 0 and interval > 0
        target = start + interval
        maxtarget = target + interval
        idx = bisect.bisect(self.times, start+interval)
        left_time = self.time(idx-1)  # We know left_time <= target < maxtarget
        if idx == len(self.times):
            right_time = maxtarget+1  # unreachable
        else:
            right_time = self.time(idx)  # We know right_time > target
        if target - left_time < right_time - target:
            if left_time > start:
                return self.images[idx-1]
            else:  # At this point we can deduce that right_time > maxtarget
                return None
        else:
            if right_time <= maxtarget:
                return self.images[idx]
            else:  # At this point we can deduce that left_time < start
                return None

    def reduce_dim(self, sample_size=1000):
        """
        Apply PCA transformation to each image of the set.
        """
        nb_images = len(self.images)
        if not sample_size:
            sample_size = nb_images
        elif 0 < sample_size <= 1:
            sample_size = int(nb_images * sample_size)
        else:
            sample_size = min(nb_images, sample_size)
        if sample_size == nb_images:
            sample = self.images
        else:
            sample = random.sample(self.images, sample_size)
        X = np.empty((sample_size, self.img_size))
        for i, img in enumerate(sample):
            X[i] = img['data']
        self.pca = PCA(copy=True).fit(X)
        for img in self.images:
            #transform returns a matrix even when given a vector
            img['data'] = self.pca.transform(img['data'])[0]
        return self.pca

    def recover_images(self, pixels):
        pixels_out = self.pca.inverse_transform(pixels)
        return pixels_out.clip(0, 1, pixels_out)

    def time(self, i):
        return self.images[i]['time']

    def __getitem__(self, item):
        return self.images[item]

    def __iter__(self):
        return iter(self.images)

    def __len__(self):
        return len(self.images)


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
        imgdim = len(images[0]['data'])
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
