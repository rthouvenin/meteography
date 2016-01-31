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
import shutil
import time

import numpy as np
import PIL
from sklearn.decomposition import PCA
import tables
from tables.nodes import filenode

MAX_PIXEL_VALUE = 255
COLOR_BANDS = ('R', 'G', 'B')
GREY_BANDS = ('L', )
PIXEL_TYPE = np.float32

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


class RawFeatures:
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.name = 'raw%dx%d' % (img_shape[1], img_shape[0])
        self.atom = tables.Atom.from_sctype(PIXEL_TYPE)
        self.nb_features = np.prod(img_shape)

    def extract(self, pixels):
        return pixels


class PCAFeatures:
    def __init__(self, pca_model):
        self.name = 'pca'
        self.atom = tables.Float32Atom()
        self.nb_features = pca_model.components_.shape[0]
        self.pca = pca_model

    def extract(self, pixels):
        return self.pca.transform([pixels])[0]

    @classmethod
    def create(cls, data, n_dims):
        #FIXME: choose an algo
        pca_model = PCA(n_components=n_dims).fit(data)
        ##self.pca = TruncatedSVD(min(300, sample_size)).fit(sample)
        return cls(pca_model)

features_extractors = {
    'raw': RawFeatures,
    'pca': PCAFeatures,
}


class FeatureSet:
    def __init__(self, root_group, features_obj=None):
        self.root = root_group
        if features_obj is None:
            fn = filenode.open_node(self.root.model, 'r')
            self.features_obj = pickle.load(fn)
            fn.close()
        else:
            self.features_obj = features_obj

    @property
    def nb_features(self):
        return self.features_obj.nb_features

    @classmethod
    def create(cls, tables_file, root, features_obj):
        group_name = features_obj.name
        group = tables_file.create_group(root, group_name)

        tables_file.create_earray(group, 'features', features_obj.atom,
                                  shape=(0, features_obj.nb_features))

        fn = filenode.new_node(tables_file, where=group, name='model')
        pickle.dump(features_obj, fn, pickle.HIGHEST_PROTOCOL)
        fn.close()

        return cls(group, features_obj)

    def remove(self):
        self.root._f_remove(recursive=True)

    def append(self, pixels):
        data = self.features_obj.extract(pixels)
        self.root.features.append([data])
        return data

    def update(self, idx, pixels):
        data = self.features_obj.extract(pixels)
        self.root.features[idx, :] = data
        return data

    def __getitem__(self, idx):
        return self.root.features[idx, :]

    def flush(self):
        return self.root.features.flush()


class ImageSet:
    # pytables descriptor for the table of images.
    table_descriptor = {
        'name': tables.StringCol(256),
        'time': tables.UIntCol(),
    }

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
        self.root = fileh.root.images
        self.table = self.root.imgset
        self.img_shape = self.table.attrs.img_shape
        self.is_grey = len(self.img_shape) == 2
        self._times = None

        self.feature_sets = {}
        for node in self.root:
            node_name = node._v_name
            if node_name not in ['imgset']:
                self.feature_sets[node_name] = FeatureSet(node)

        if 'pca' in self.feature_sets:
            self.is_reduced = True
            self.img_size = self.feature_sets['pca'].nb_features
        else:
            self.is_reduced = False
            self.img_size = np.prod(self.img_shape)

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
            group = fp.create_group('/', 'images')
            table = fp.create_table(group, 'imgset', cls.table_descriptor)
            table.attrs.img_shape = img_shape
            table.cols.time.create_csindex()

            imgset = cls(fp)
            extractor = RawFeatures(img_shape)
            imgset.add_feature_set(extractor)  # FIXME rm?

            return imgset
        except Exception:
            #Avoid an orphan open file in case of a problem
            if fp is not thefile:
                fp.close()
            raise

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def add_feature_set(self, extractor=None, name=None, params=None, **kwargs):
        if extractor is None:
            if name not in features_extractors:
                raise ValueError("No features extractor named %s" % name)
            feature_class = features_extractors[name]
            instance_params = params if params is not None else kwargs
            extractor = feature_class(**instance_params)

        if extractor.name not in self.feature_sets:
            logger.info("The features set already exists, NOT adding it")
            feature_set = FeatureSet.create(self.fileh, self.root, extractor)
            self.feature_sets[extractor.name] = feature_set

            # Extract the features of existing images and append them
            for img in self:
                feature_set.append(img['pixels'])
            feature_set.flush()

        return self.feature_sets[extractor.name]

    def get_feature_set(self, feature_set):
        if feature_set not in self.feature_sets:
            raise ValueError("There is no features set named %s" % feature_set)
        return self.feature_sets[feature_set]

    def remove_feature_set(self, feature_set):
        if feature_set not in self.feature_sets:
            raise ValueError("There is no features set named %s" % feature_set)
        self.feature_sets[feature_set].remove()
        del self.feature_sets[feature_set]

    def _img_dict(self, name, time, pixels, img_id):
        return {
            'name': name,
            'time': time,
            'pixels': pixels,
            'id': img_id
        }

    def _img_from_row(self, row, feature_set=None):
        """
        Create from a row a dictionary with the details of the image in that
        row. If `ret_reduced` is True and the ImageSet was reduced, the key
        'pixels' will contain the PCA-transformed data.
        """
        if feature_set is None:
            pixels = self.pixels_from_file(row['name'])
        else:
            feature_set = self.get_feature_set(feature_set)
            pixels = feature_set[row.nrow]

        img = self._img_dict(row['name'], row['time'], pixels, row.nrow)
        return img

    def get_image(self, t, feature_set=None):
        """
        Returns as a dictionary the details of the image taken at time `t`.
        If `ret_reduced` is True and the ImageSet was reduced, the key 'pixels'
        will contain the PCA-transformed data.
        """
        rows = self.table.where('time == t')
        try:
            row = next(rows)
            return self._img_from_row(row, feature_set)
        except StopIteration:
            raise ValueError("Image with time=%d does not exist" % t)

    def _realpixels(self, idx):
        rows = self.table.itersequence(idx)
        return np.array([self.pixels_from_file(r['name']) for r in rows])

    def get_pixels_at(self, idx, feature_set=None):
        """
        Retrieves the pixels of one or more images referenced by indices.

        Parameters
        ----------
        idx : integral number or sequence of integral numbers
            The indices of the images
        ret_reduced : bool
            If True and the ImageSet was reduced, the PCA-transformed data
            is returned
        """
        if feature_set is None:
            source = self._realpixels
        else:
            feature_set = self.get_feature_set(feature_set)
            source = lambda i: feature_set[i]

        #numpy scalars also have a __getitem__ ...
        if hasattr(idx, '__getitem__') and not np.isscalar(idx):
            result = source(idx)
        else:
            result = source((idx, ))
        return result

    def __iter__(self):
        """
        Create a generator on the images, returned as dictionaries.
        """
        rows = self.table.iterrows()
        for row in rows:
            yield self._img_from_row(row)

    def __len__(self):
        """Return the number of images in the set."""
        return self.table.nrows

    def _add_image(self, name, img_time, pixels, ret_features=None):
        img_id = self.table.nrows
        ret_pixels = pixels
        if ret_features is not None:
            ret_features = self.get_feature_set(ret_features)

        # If there is already an image for this time, update it
        existing = self.table.where('time == img_time')
        existing = next(existing, None)
        if existing is not None:
            img_id = existing.nrow
            for feature_set in self.feature_sets.values():
                appended = feature_set.update(img_id, pixels)
                if feature_set is ret_features:
                    ret_pixels = appended
            return self._img_dict(name, img_time, ret_pixels, img_id)

        # Otherwise, add it...
        else:
            # ...to the table...
            row = self.table.row
            row['name'] = name
            row['time'] = img_time
            row.append()

            # ...and each feature set
            for feature_set in self.feature_sets.values():
                appended = feature_set.append(pixels)
                if feature_set is ret_features:
                    ret_pixels = appended

            # update or invalidate times cache
            if self._times is not None and img_time > self._times[-1]:
                self._times.append(img_time)
            else:
                self._times = None

            return self._img_dict(name, img_time, ret_pixels, img_id)

    def pixels_from_file(self, imgfile):
        """
        Extract the data from an image file located at `imgfile`.

        Return
        ------
        A one-dimension numpy array of floats (c.f. PIXEL_TYPE) in [0, 1].
        If the set is working with grey-scale images, the size of the returned
        array is the number of pixels, otherwise 3 times as much. The elements
        are ordered band by band and line by line:
            [r(0,0), g(0,0), b(0,0), r(0,1), g(0,1), b(0,1), ...]
        """
        with open(imgfile, 'rb') as fp:
            img = PIL.Image.open(fp)
            img_shape = img.size[1], img.size[0]
            if img_shape != self.img_shape[:2]:
                raise ValueError("The image has shape %s instead of %s"
                                 % (img_shape, self.img_shape[:2]))

            # Convert the image to greyscale or RGB if necessary
            expected_bands = GREY_BANDS if self.is_grey else COLOR_BANDS
            img_bands = img.getbands()
            if img_bands != expected_bands:
                img = img.convert(''.join(expected_bands))
            data = np.array(img, dtype=PIXEL_TYPE) / MAX_PIXEL_VALUE

            # Flatten the data in case of RGB tuples
            if data.ndim > 1:
                data = data.flatten()

            return data

    def _add_from_file(self, imgfile, name_parser, ret_features=None):
        """
        `add_from_file` without flushing the table.
        """
        data = self.pixels_from_file(imgfile)
        img_time = name_parser(imgfile)
        return self._add_image(imgfile, img_time, data, ret_features)

    def add_from_file(self, imgfile, name_parser=parse_timestamp,
                      ret_features=None):
        """
        Add an image to the set.

        Parameters
        ----------
        imgfile : str
            The file name of the image
        name_parser : callable
            A function that takes an absolute filename as argument and
            returns the timestamp of when the photo was taken
        ret_reduced : bool
            Whether the returned image should come with reduced pixels
            (if available)

        Return
        ------
        int : The time (Unix epoch) when the image was taken, that can be used
            to identify it
        """
        img = self._add_from_file(imgfile, name_parser, ret_features)
        self.table.flush()
        return img

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
                try:
                    self._add_from_file(fullname, name_parser)
                except Exception as e:
                    logger.warn("Ignoring file %s: %s" % (fullname, e))
        self.table.flush()

    def sample(self, sample_size):
        """
        Pick randomly `sample_size` images and return their pixels as a single
        ndarray of shape (`sample_size`, `img_size`).

        Parameters
        ----------
        sample_size : number or None
            Indicates how many images should be used to compute the PCA
            reduction. If 0 or None, all the images of the set are used. If it
            is a float in ]0, 1], it indicates the proportion of images to be
            used. If greater than 1, it indicates the maximum number of images
            to use.
        """
        # Compute the sample size and create it
        nb_images = self.table.nrows
        if not sample_size:
            sample_size = nb_images
        elif 0 < sample_size <= 1:
            sample_size = int(nb_images * sample_size)
        else:
            sample_size = min(nb_images, sample_size)

        # Create the sample
        if sample_size == nb_images:
            sample = self.get_pixels_at(range(len(self)))
        else:
            index = random.sample(xrange(self.table.nrows), sample_size)
            sample = self.get_pixels_at(index)
        return sample

    def reduce_dim(self, sample_size=1000, n_dims=.99):
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
        sample = self.sample(sample_size)

        # Compute PCA components from the sample.
        extractor = PCAFeatures.create(sample, n_dims)
        self.img_size = extractor.nb_features

        # Create the new FeatureSet
        if self.is_reduced:
            self.remove_feature_set('pca')
        self.add_feature_set(extractor)

        self.is_reduced = True

    def find_closest(self, start, interval, feature_set=None):
        """
        Search for the image with its time attribute the closest to
        `start-interval`, constrained in `[start-2*interval, start[`.
        This is a binary search in O(log n), except the first time it is called

        Parameters
        ----------
        start : int
        interval : int
        ret_reduced : bool
            Whether the returned image should come with reduced pixels
            (if available)

        Return
        ------
        The image (a `dict`) or `None` if none is found
        """
        assert start >= 0 and interval > 0
        target = start - interval
        mintarget = target - interval
        if self._times is None:
            self._times = list(self.table.cols.time[:])
            self._times.sort()

        idx = bisect.bisect_left(self._times, target)
        if idx == len(self._times):
            right_time = start  # unreachable
        else:
            right_time = self._times[idx]  # right_time > target > mintarget
        if idx == 0:
            left_time = mintarget-1  # unreachable
        else:
            left_time = self._times[idx-1]  # left_time <= target

        if target - left_time < right_time - target:
            if left_time >= mintarget:
                return self.get_image(left_time, feature_set)
            else:  # At this point we can deduce that right_time >= start
                return None
        else:
            if right_time < start:
                return self.get_image(right_time, feature_set)
            else:  # At this point we can deduce that left_time < mintarget
                return None


class DataSet:
    def __init__(self, fileh, imageset):
        """
        Build a DataSet from the images of `imageset`.
        The input and target data are built, but not split into training,
        validation and test sets (use `split` for that).

        Parameters
        ----------
        imageset : ImageSet
            The ImageSet containing all the source images.
        """
        self._reset(fileh, imageset)

    def _reset(self, fileh, imageset):
        self.fileh = fileh
        self.imgset = imageset
        self.img_shape = imageset.img_shape

    @property
    def is_reduced(self):
        return self.imgset.is_reduced

    @classmethod
    def create(cls, imgset, thefile=None):
        """
        Create in `thefile` a pytables node 'examples' to be used for datasets.
        Create a DataSet backed by this file and with `imgset` as image source.

        Parameters
        ----------
        imgset : ImageSet
            The image source to be used to construct learning examples
        thefile : str or pytables file descriptor or None
            The name of the file to create, or a file descriptor already
            opened. If the name of an existing file is given, it will be
            overwritten.
            If None, the same file as the ImageSet's one will be used
        """
        if thefile is None:
            fp = imgset.fileh
        elif not hasattr(thefile, 'create_table'):
            fp = tables.open_file(thefile, mode='w')
        else:
            fp = thefile

        try:
            fp.create_group('/', 'examples')
        except Exception as e:
            #Avoid an orphan open file in case of a problem
            if fp is not thefile:
                fp.close()
            raise e
        return cls(fp, imgset)

    @classmethod
    def open(cls, imageset, thefile=None):
        """
        Instantiate a DataSet backed by `thefile` and imageset.

        Parameters
        ----------
        imageset : ImageSet or file argument
            The image source to be used to construct learning examples, or a
            file that contains an ImageSet
        thefile : str or pytables file descriptor or None
            The name of the file to open, or a pytables file descriptor already
            opened. If None, the one of the ImageSet will be used.
        """
        if not hasattr(imageset, 'find_closest'):
            imageset = ImageSet.open(imageset)

        if thefile is None:
            fp = imageset.fileh
        elif not hasattr(thefile, 'create_table'):
            fp = tables.open_file(thefile, mode='a')
        else:
            fp = thefile
        return cls(fp, imageset)

    def close(self):
        self.imgset.close()
        return self.fileh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _dictify(self, fuzzy_img):
        """
        Convert fussy_img to dictionary representation.
        `fuzzy_img` can be a time (int) or dictionary.
        """
        if not hasattr(fuzzy_img, '__getitem__'):
            #fuzzy_img is in fact a time, retrieve the image
            img = self.imgset.get_image(fuzzy_img)
        else:
            img = fuzzy_img
        return img

    def _nodify(self, fuzzy_node):
        "Retrieve a set of examples by name, if `fuzzy_node` is not a node."
        if not hasattr(fuzzy_node, '_v_attrs'):
            ex_set = self.fileh.get_node(self.fileh.root.examples, fuzzy_node)
        else:
            ex_set = fuzzy_node
        return ex_set

    def init_set(self, name, intervals, feature_set=None):
        """
        Create a new node in '/examples' of the given `name`, that will store
        the input and output of examples for the given parameters.

        Parameters
        ----------
        name : string or None
            The name of the set (for future reference).
        intervals : a sequence or None
            The amount of time between each image of an example. The last
            element is the amount of time between the last image of the input
            and the output image. If None, the sequence is created from the
            values of `hist_len`, `interval` and `future_time`.

        Return
        ------
        pytables node : the created node.
        """
        if feature_set is None:
            feature_set = self.imgset.feature_sets.keys()[0]  # FIXME
        self.delete_set(name)
        ex_group = self.fileh.root.examples
        ex_set = self.fileh.create_group(ex_group, name)
        ex_set._v_attrs.intervals = intervals
        ex_set._v_attrs.features = feature_set
        self._create_set_arrays(ex_set, 'img_refs', 'input', 'output')
        return ex_set

    def _create_set_arrays(self, ex_set, refs_name, in_name, out_name):
        fh = self.fileh
        hist_len = len(ex_set._v_attrs.intervals)
        if self.is_reduced:
            imgdim = self.imgset.img_size
        else:
            imgdim = np.prod(self.img_shape)
        nb_feat = (imgdim+1) * hist_len
        pixel_atom = tables.Atom.from_sctype(PIXEL_TYPE)

        if refs_name is not None:
            fh.create_earray(ex_set, refs_name, atom=tables.IntAtom(),
                             shape=(0, hist_len+1))
        if in_name is not None:
            fh.create_earray(ex_set, in_name, atom=pixel_atom,
                             shape=(0, nb_feat))
        if out_name is not None:
            fh.create_earray(ex_set, out_name, atom=pixel_atom,
                             shape=(0, imgdim))
        self.fileh.flush()

    def get_set(self, name):
        return self._nodify(name)

    def make_set(self, name, intervals, feature_set=None):
        """
        Constructs a set of training examples from all the available images.
        If a node with the same name already exists, it is overwritten.
        The meaning of the parameters is the same as for `init_set`
        """
        #Create pytables group and arrays
        newset = self.init_set(name, intervals, feature_set)
        intervals = newset._v_attrs.intervals

        #Find the representations of the data points
        examples = self._find_examples(intervals, feature_set)

        img_refs = newset.img_refs
        input_data = newset.input
        output_data = newset.output
        nb_feat = input_data.shape[1]
        #Copy the data into the arrays
        input_row = np.empty((nb_feat,), PIXEL_TYPE)
        input_rows = [input_row]
        for i, example in enumerate(examples):
            ids = [img['id'] for img in example[0]]
            ids.append(example[1]['id'])
            img_refs.append([ids])
            self._get_input_data(example[0], example[1]['time'], input_row)
            input_data.append(input_rows)
            output_data.append([example[1]['pixels']])

        img_refs.flush()
        input_data.flush()
        output_data.flush()
        return newset

    def delete_set(self, name):
        """
        Delete a set of training examples.
        Silent ignore if the set does not exist.
        """
        ex_group = self.fileh.root.examples
        if name in ex_group:
            self.fileh.remove_node(ex_group, name, recursive=True)
            self.fileh.flush()

    def _find_examples(self, intervals, feature_set):
        """
        Generator of tuples (input_images, output_value) to be used for
        training or validation.

        Returns
        -------
        list(tuple(list(filedict), filedict)) : a list of tuples
            representing examples, as returned by `_find_example`
        """
        assert all(intervals)
        for i, img in enumerate(self.imgset):
            example = self._find_example(img, intervals, feature_set)
            if example is not None:
                yield example

    def _find_example(self, img, intervals, feature_set):
        """
        Try to find a single example that has `img` as a target

        Parameters
        ----------
        img : int or dict
            the time when the image was taken, or a dict representing the image
            such as returned by `ImageSet.get_image`
        intervals: sequence
            c.f. `init_set`

        Return
        ------
        tuple(list(filedict), filedict) : a tuple (or None)
            (input_images, output_value) where input_images is a list of images
            to be used as the input of an example, and output_value is the
            target (expected prediction) of the example
        """
        img = self._dictify(img)
        #Search for input images that match the target
        images = self._find_sequence(img, intervals, feature_set)

        #If we could find all of them, return an example
        hist_len = len(intervals)
        if len(images) == hist_len+1:
            inputs = images[:hist_len]
            target = images[-1]
            return inputs, target
        return None

    def _find_sequence(self, img, intervals, feature_set):
        """
        Find a sequence of images ending with `img` and matching `intervals`
        """
        assert img is not None
        images = [img]
        for j, interval in enumerate(reversed(intervals)):
            prev_time = images[j]['time']
            img_j = self.imgset.find_closest(prev_time, interval, feature_set)
            if img_j:
                images.append(img_j)
            else:
                break
        images.reverse()
        return images

    def make_input(self, ex_set, fuzzy_img, target_time=None):
        """
        Try to construct the input of an example such that `img` is the last
        image.

        Parameters
        ----------
        ex_set : pytables node or str
            A set of examples such as returned by `init_set`, or the name of
            such a set
        img : dict or int
            An image such as returned by `ImageSet.get_image`, or the time when
            the image was taken
        target_time : int
            How far in the future the prediction of the returned input will be

        Returns
        -------
        array : the feature vector
        """
        img = self._dictify(fuzzy_img)
        ex_set = self._nodify(ex_set)
        intervals = ex_set._v_attrs.intervals[:-1]
        feature_set = ex_set._v_attrs.features
        if target_time is None:
            target_time = ex_set._v_attrs.intervals[-1]

        images = self._find_sequence(img, intervals, feature_set)
        if len(images) == len(intervals) + 1:
            prediction_time = target_time + images[-1]['time']
            return self._get_input_data(images, prediction_time)
        return None

    def _get_input_data(self, ex_input, target_time, ex_data=None):
        """
        Write the input pixels and features of `example` in array `ex_data`.
        If `ex_data` is None, a new array is created.
        """
        imgdim = len(ex_input[0]['pixels'])
        hist_len = len(ex_input)
        if ex_data is None:
            nb_feat = (imgdim+1) * hist_len
            ex_data = np.empty((nb_feat,), PIXEL_TYPE)

        for j, img in enumerate(ex_input):
            ex_data[imgdim*j:imgdim*(j+1)] = img['pixels']
            ex_data[j-hist_len] = target_time - img['time']
        return ex_data

    def add_image(self, imgfile, ex_set=None):
        """
        Add an image to the underlying ImageSet, and creates a new example
        in each example set using this image as expected prediction.

        Parameters
        ----------
        imgfile : str
            Filename of the image to add
        setname : str
            Name of the dataset where the example should be added
        """
        feature_set = 'pca' if self.is_reduced else None
        img = self.imgset.add_from_file(imgfile, ret_features=feature_set)

        if ex_set is not None:
            ex_sets = [self._nodify(ex_set)]
        else:
            ex_sets = iter(self.fileh.root.examples)

        for ex_set in ex_sets:
            intervals = ex_set._v_attrs.intervals
            feature_set = ex_set._v_attrs.features
            example = self._find_example(img, intervals, feature_set)

            if example is not None:
                imgs_in, img_out = example
                ids = [im['id'] for im in imgs_in]
                ids.append(img_out['id'])
                ex_set.img_refs.append([ids])
                input_row = self._get_input_data(imgs_in, img_out['time'])
                ex_set.input.append([input_row])
                ex_set.output.append([img_out['pixels']])
                ex_set.img_refs.flush()
                ex_set.input.flush()
                ex_set.output.flush()
        return img

    def reduce_dim(self):
        """
        Perform dimensionality reduction on the underlying ImageSet, and
        re-compute the input and output arrays of all the sets of examples.
        Note that re-computing the reduction after some images were added may
        INCREASE the dimensionality (but would improve the sampling)
        """
        self.imgset.reduce_dim()
        for ex_set in self.fileh.root.examples:
            logger.info("Applying the reduction to set %s", ex_set._v_name)
            ex_set._v_attrs.features = 'pca'
            self._recompute_set(ex_set)

    def _recompute_set(self, ex_set):
        """
        Re-create the input and output arrays of `ex_set`.
        """
        hist_len = len(ex_set._v_attrs.intervals)
        feature_set = ex_set._v_attrs.features
        self._create_set_arrays(ex_set, None, 'newinput', 'newoutput')
        for refs_row in ex_set.img_refs:
            pixels_row = self.imgset.get_pixels_at(refs_row, feature_set)
            newfeatures = pixels_row[:-1].flatten()
            oldfeatures = ex_set.input[ex_set.img_refs.nrow, -hist_len:]
            ex_set.newinput.append([np.hstack([newfeatures, oldfeatures])])
            ex_set.newoutput.append([pixels_row[-1]])
        ex_set.input.remove()
        ex_set.output.remove()
        ex_set.newinput.rename('input')
        ex_set.newoutput.rename('output')

    def input_img(self, ex_set, i, j):
        """
        Return the pixels the `j`th image from example `i` in the input data,
        with its original shape (directly displayable by matplotlib)
        """
        ex_set = self._nodify(ex_set)
        imgdim = ex_set.output.shape[1]
        img = ex_set.input[i, j*imgdim:(j+1)*imgdim]
        return img.reshape(self.img_shape)

    def output_img(self, ex_set, i):
        """
        Return the pixels the `i`th output image,
        with its original shape (directly displayable by matplotlib)
        """
        ex_set = self._nodify(ex_set)
        img_ref = ex_set.img_refs[i][-1]
        raw_data = self.imgset.get_pixels_at(img_ref, None)
        return raw_data.reshape(self.img_shape)

    def repack(self):
        """
        Recreate the HDF5 backing this dataset and underlying imageset.
        As HDF5 does not free the space after removing nodes from a file, it is
        necessary to re-create the entire file if one wants to reclaim the
        space, which is of course an expensive operation on large files.

        You may want to call repack after these operations:
         - delete_set
         - reduce_dim

        Notes
        -----
         - If the DataSet was opened or created from a file pointer, this
        pointer will be closed and should not be used after the repack call
         - This will repack the entire file, including the ImageSet data and
        any other data that may be there from other sources.
        """
        # Copy the file over itself
        old_name = self.fileh.filename
        temp_name = old_name + '.temp'
        self.fileh.copy_file(temp_name)
        self.fileh.close()
        shutil.move(temp_name, old_name)

        # Re-initialize the file pointer and the ImageSet instance
        if self.imgset.fileh is self.fileh:
            new_imgset = ImageSet.open(old_name)
            new_fileh = new_imgset.fileh
        else:
            new_imgset = self.imgset
            new_fileh = tables.open_file(old_name, mode='a')
        self._reset(new_fileh, new_imgset)
