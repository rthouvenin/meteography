# -*- coding: utf-8 -*-
"""
Classes and associated helper functions to create and manipulate the data sets
used in the machine learning machinery
TODO: doc and tests
TODO: log ignored files
"""

import bisect
import os.path
import pickle

import numpy as np
import PIL
from sklearn.decomposition import PCA

MAX_PIXEL_VALUE = 255
COLOR_BANDS = ('R', 'G', 'B')
GREY_BANDS = ('L', )


class ImageSet:
    def __init__(self, images):
        assert images
        self.images = sorted(images, key=lambda x: x['time'])
        self.times = [i['time'] for i in self.images]
        self.img_shape = images[0]['shape']

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

    def find_datapoints(self, hist_len, interval, future_time):
        """
        Build a list of tuples (input_images, output_value).

        Parameters
        ----------
        hist_len : int (strictly greater than 0)
            The number of input images in each data point
        interval : int (strictly greater than 0)
            The approximate (+/- 100%) time interval between each input image
        future_time : int (strictly greater than 0)
            The approximate (+/- 100%) time interval between the last input
            image and the output value
        """
        assert hist_len > 0 and interval > 0 and future_time > 0
        data_points = []
        for i, img in enumerate(self):
            images_i = [img]
            for j in range(1, hist_len):
                prev_time = images_i[j-1]['time']
                img_j = self.find_closest(prev_time, interval)
                if img_j:
                    images_i.append(img_j)
                else:
                    break
            if len(images_i) == hist_len:
                latest_time = images_i[-1]['time']
                target = self.find_closest(latest_time, future_time)
                if target:
                    data_points.append((images_i, target))
        return data_points

    @classmethod
    def make(cls, dirpath, greyscale=False):
        """
        Build a ImageSet from the images located in the directory `dirpath`.
        """
        filenames = [os.path.join(dirpath, f) for f in os.listdir(dirpath)]
        files = map(parse_filename, filenames)
        #Filter out junk
        files = [f for f in files if 'error' not in f]
        imgshape = files[0]['shape']
        files = [f for f in files if f['shape'] == imgshape]
        return cls(map(lambda f: extract_image(f, greyscale), files))

    def time(self, i):
        return self.images[i]['time']

    def __getitem__(self, item):
        return self.images[item]

    def __iter__(self):
        return iter(self.images)


class DataSet:
    def __init__(self, img_shape, indata, outdata):
        self.img_shape = img_shape
        self.input_data = indata
        self.output_data = outdata
        self.img_size = np.prod(self.img_shape)
        self.history_len = indata.shape[1] % self.img_size
        self.is_split = False

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

    def reduce_dim(self, reduce_output=True):
        """
        Apply PCA transformation to each input image of the dataset, and to
        output images as well if `reduce_output` is True.
        When the dataset was split, the reduction matrix is computed on the
        training image (but the whole dataset is then reduced).
        """
        Xtrain = self.training_set()[0]
        #Approx. of all training images: the first image of each training input
        X = Xtrain[:, :self.img_size]
        self.pca = PCA(copy=True).fit(X)
        self.input_data = self.reduce_input(self.input_data)
        if reduce_output:
            self.output_data = self.pca.transform(self.output_data)
        if self.is_split:
            train_size = len(Xtrain)
            valid_size = len(self.valid_input)
            self.__dispatch_data(train_size, valid_size, reduce_output)
        return self

    def reduce_input(self, input_data):
        """
        Uses the reduction matrix computed by `reduce_dim` to transform the
        given `input_data`. It is an error to call this method if `reduce_dim`
        was not called.
        """
        if len(input_data):
            X = input_data[:, :-self.history_len]
            X = X.reshape((-1, self.img_size))
            reduced = self.pca.transform(X)
            reduced = reduced.reshape((len(input_data), -1))
            return np.hstack((reduced, input_data[:, -self.history_len:]))
        return input_data

    def recover_output(self, output_data):
        return self.pca.inverse_transform(output_data)

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

    @classmethod
    def make(cls, imageset, hist_len=3, interval=600, future_time=1800,
             greyscale=False):
        """
        Build a DataSet from the images of `imageset`.

        The input and target data are built, but not split into training,
        validation and test sets (use `split` for that). When the source is
        a directory, the file names without the extension should be the Epoch
        when the photo was taken.
        All the files are attempted to be read, the ones cannot be read as
        an image or whose time cannot be read are ignored.
        All the images are expected to have the same dimensions.

        Parameters
        ----------
        imageset : str or ImageSet
            The directory or ImageSet containing all the source images.
        hist_len : int
            The number of images to include in the input data
        interval : int
            The number of seconds between each input image
        future_time : int
            The number of seconds between the latest input image
            and the target image
        greyscale : bool
            Whether to work on greyscale images (True) or color images (False)
        """
        if not hasattr(imageset, 'find_datapoints'):
            imageset = ImageSet.make(imageset, greyscale)
        #Make actual input and output data
        X, y = cls.make_datapoints(imageset, hist_len, interval, future_time)
        return cls(imageset.img_shape, X, y)

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

    @staticmethod
    def make_datapoints(images, hist_len, interval, future_time):
        """
        Auxiliary method of `DataSet.make` to create the data points from the
        ImageSet `images`.
        """
        #Find data points
        data_points = images.find_datapoints(hist_len, interval, future_time)
        #Copy the data into numpy arrays
        imgdim = np.prod(images[0]['shape'])
        nb_ex = len(data_points)
        nb_feat = (imgdim+1) * hist_len
        input_data = np.ndarray((nb_ex, nb_feat), dtype=np.float32)
        output_data = np.ndarray((nb_ex, imgdim), dtype=np.float32)
        for i, data_point in enumerate(data_points):
            example, target = data_point
            for j, img in enumerate(example):
                input_data[i, imgdim*j:imgdim*(j+1)] = img['data']
                input_data[i, j-hist_len] = target['time'] - img['time']
            output_data[i] = target['data']
        return input_data, output_data


def parse_filename(filename):
    """
    Open `filename` as an image and read some metadata.
    """
    filedict = {'fullpath': filename}
    fp = None
    try:
        #PIL will take care of closing the file when loading the data
        fp = open(filename, 'rb')
        img = PIL.Image.open(fp)
        filedict['pil_img'] = img
        filedict['shape'] = img.size[1], img.size[0]
        basename = os.path.basename(filename)
        strtime = basename[:basename.index('.')]
        filedict['time'] = int(strtime)
    except Exception as e:
        filedict['error'] = e
        if fp:
            fp.close()
    return filedict


def extract_image(file_dict, grayscale=False):
    """
    Read the pixels of the image in `filedict` and store them in a 'data'
    attribute as a flat array with the value of the pixels in [0,1]
    """
    expected_bands = GREY_BANDS if grayscale else COLOR_BANDS
    bands = file_dict['pil_img'].getbands()
    img = file_dict['pil_img']
    if bands != expected_bands:
        img = img.convert(''.join(expected_bands))
    raw_data = img.getdata()
    data = np.asarray(raw_data, dtype=np.float32) / MAX_PIXEL_VALUE
    #Flatten the data in case of RGB tuples
    if len(data.shape) > 1:
        data = data.flatten()
    file_dict['data'] = data
    if not grayscale:
        file_dict['shape'] += (3, )
    return file_dict
