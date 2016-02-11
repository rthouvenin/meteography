"""
Classes and associated helper functions to extract and manipulate the features
from the input data
"""
import cPickle as pickle

import numpy as np
import scipy.misc
from sklearn.decomposition import PCA
from sklearn.neural_network.rbm import BernoulliRBM
import tables
from tables.nodes import filenode

COLOR_BANDS = ('R', 'G', 'B')
GREY_BANDS = ('L', )
MAX_PIXEL_VALUE = 255
PIXEL_TYPE = np.float32


def accepts_one_row(func):
    """
    Method decorator for functions that process a 2-d array.
    Adds the possibility of being called with a 1-d array (a single row) and
    returning a 1-d array (first row of the result)
    """
    def process(self, arr):
        if arr.ndim == 1:
            return func(self, [arr])[0]
        else:
            return func(self, arr)
    return process


class FeaturesExtractor(object):
    """
    Base class of features extractors

    Attributes
    ----------
    name : str
        An identifier of the extractor and its parameters
    atom :
        The Pytable atom used to store a single feature
    nb_features :
        The number of features that this extractor generates
    """
    def __init__(self, name, atom, nb_features):
        self.name = name
        self.atom = atom
        self.nb_features = nb_features


class RawFeatures(FeaturesExtractor):
    def __init__(self, img_shape, src_shape):
        """
        Features extractor that simply resizes the image and
        extracts the pixels.

        Init parameters
        ---------------
        img_shape : tuple
            A 2-tuple (for greyscale images) or 3-tuple (for color images)
            that is the target shape to resize to: (height, width, nb_bands)
        src_shape : tuple
            A 2-tuple (for greyscale images) or 3-tuple (for color images)
            that is the source shape to resize from: (height, width, nb_bands)
        """
        self.dest_bands = img_shape[2] if len(img_shape) == 3 else 1
        self.dest_size = tuple(reversed(img_shape[:2]))
        self.src_shape = src_shape
        self.src_bands = src_shape[2] if len(src_shape) == 3 else 1
        self.src_size = tuple(reversed(src_shape[:2]))

        name = 'raw%dx%d' % (img_shape[1], img_shape[0])
        atom = tables.Atom.from_sctype(PIXEL_TYPE)
        nb_features = np.prod(img_shape)
        super(RawFeatures, self).__init__(name, atom, nb_features)

    def _extract_single(self, pixels):
        if self.dest_size != self.src_size or self.dest_bands != self.src_bands:
            img = scipy.misc.toimage(pixels.reshape(self.src_shape))

            if self.dest_size != self.src_size:
                img = img.resize(self.dest_size)

            if self.dest_bands != self.src_bands:
                if self.dest_bands == 3:
                    mode = ''.join(COLOR_BANDS)
                else:
                    mode = ''.join(GREY_BANDS)
                img = img.convert(mode)

            pixels = np.array(img, dtype=PIXEL_TYPE) / MAX_PIXEL_VALUE
            pixels = pixels.flatten()
        return pixels

    def extract(self, pixels):
        """
        Resize the image

        Parameters
        ----------
        pixels : array
            1-d array of floats representing the image pixels, scaled to [0, 1]
            or 2-d array representing a sequence of images
        """
        if pixels.ndim == 1:
            return self._extract_single(pixels)
        else:
            res = np.empty((len(pixels), self.nb_features), dtype=PIXEL_TYPE)
            for i, img_pixels in enumerate(pixels):
                res[i] = self._extract_single(img_pixels)
            return res


class PCAFeatures(FeaturesExtractor):
    def __init__(self, pca_model):
        """
        Features extractor that computes the PCA components of the images.
        The name is a constant: it does not support multiple instances with
        different PCA parameters.

        Init parameters
        ---------------
        pca_model : sklearn.decomposition.PCA instance
            The model to use to compute PCA components
        """
        self.pca = pca_model
        name = 'pca'
        atom = tables.Float32Atom()
        nb_features = pca_model.components_.shape[0]
        super(PCAFeatures, self).__init__(name, atom, nb_features)

    @accepts_one_row
    def extract(self, pixels):
        return self.pca.transform(pixels)

    @classmethod
    def create(cls, data, n_dims=.99):
        """
        Computes the PCA components from the given data and return a
        PCAFeatures instance based on these components

        Parameters
        ----------
        data : numpy array
            The data to use to compute the PCA components
        n_dims : number or None
            Number of components to keep.
            If n_components is not set all components are kept.
            If 0 < n_components < 1, select the number of components such that
            the amount of variance that needs to be explained is greater than
            the percentage specified by n_components.
        """
        pca_model = PCA(n_components=n_dims).fit(data)
        return cls(pca_model)


class RBMFeatures(FeaturesExtractor):
    def __init__(self, rbm_model):
        """
        Features extractor that uses a Rietz-Boltzmann Machine
        to compute the representation images.

        Init Parameters
        ---------------
        rbm_model : sklearn.neural_network.rbm.BernoulliRBM instance
            The model to use to compute the components
        """
        self.rbm = rbm_model
        name = 'rbm'
        atom = tables.Float32Atom()
        nb_features = rbm_model.n_components
        super(RBMFeatures, self).__init__(name, atom, nb_features)

    @accepts_one_row
    def extract(self, pixels):
        return self.rbm.transform(pixels)

    @classmethod
    def create(cls, data, n_dims=144):
        """
        Computes the components from the given data and return a
        RBMFeatures instance based on these components

        Parameters
        ----------
        data : numpy array
            The data to use to compute the PCA components
        n_dims : int
            Number of components to keep.
        """
        rbm_model = BernoulliRBM(n_components=n_dims).fit(data)
        return cls(rbm_model)


def extractor_cached(func):
    """A memoizer for methods of FeatureSet that return a feature extractor"""
    cache = {}

    def memoizer(*args, **kwargs):
        feature_set = args[0]
        filename = feature_set.root._v_file.filename
        set_name = feature_set.root._v_pathname
        key = '%s-%s' % (filename, set_name)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoizer


class FeatureSet(object):
    """
    Represents a dataset containing the features of a set of images.
    It is a wrapper of a Pytables group where the data is stored.

    Attributes
    ----------
    nb_features : int
        the number of features stored for each image
    name : str
        the name of this set, unique in a ImageSet
    """
    def __init__(self, where, extractor=None):
        """
        Open a FeatureSet from an existing Pytables group, or create one.

        Init parameters
        ---------------
        where : Pytables group
            Where in the pytables file the set is stored. If `extractor` is
            not provided, the set is read from the existing `where` node. If it
            is provided, the set is created in `extractor.name` under `where`.
        extractor : FeaturesExtractor or None (optional)
            The extractor to use to add data to this feature set.
            If None, it is read from the data stored in the Pytables group
        """
        if extractor is None:
            self.root = where
            self.extractor = self._get_model()

        else:
            fp = where._v_file
            self.root = fp.create_group(where, extractor.name)
            fp.create_earray(self.root, 'features', extractor.atom,
                             shape=(0, extractor.nb_features))

            self.extractor = self._store_model(extractor)

    @property
    def nb_features(self):
        return self.extractor.nb_features

    @property
    def name(self):
        return self.extractor.name

    def _store_model(self, extractor):
        fn = filenode.new_node(self.root._v_file,
                               where=self.root, name='model')
        pickle.dump(extractor, fn, pickle.HIGHEST_PROTOCOL)
        fn.close()
        return extractor

    @extractor_cached
    def _get_model(self):
        fn = filenode.open_node(self.root.model, 'r')
        extractor = pickle.load(fn)
        fn.close()
        return extractor

    def remove(self):
        "Delete the associated Pytables group"
        self.root._f_remove(recursive=True)

    def append(self, pixels):
        "Extract features from the given data and add them to the set"
        data = self.extractor.extract(pixels)
        self.root.features.append([data])
        return data

    def update(self, idx, pixels):
        "Extract features from the given data and update the rows at `idx`"
        data = self.extractor.extract(pixels)
        self.root.features[idx, :] = data
        return data

    def __getitem__(self, idx):
        "Return the data at row(s) `idx`"
        return self.root.features[idx, :]

    def __len__(self):
        return self.root.features.shape[0]

    def flush(self):
        "Flush on disk any data that is still in buffers"
        return self.root.features.flush()
