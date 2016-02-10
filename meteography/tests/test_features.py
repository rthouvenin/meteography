# -*- coding: utf-8 -*-
import numpy as np

from meteography.features import RawFeatures

IMG_SIZE = 20


class TestRawFeatures:
    def test_extract_samesize(self):
        shape = (IMG_SIZE, IMG_SIZE)
        data = np.random.rand(IMG_SIZE * IMG_SIZE)
        extractor = RawFeatures(shape, shape)
        features = extractor.extract(data)
        assert(features is data)

    def test_extract_halfsizetogrey(self):
        src_shape = (IMG_SIZE, IMG_SIZE, 3)
        dest_shape = (IMG_SIZE / 2, IMG_SIZE / 2)
        data = np.random.rand(IMG_SIZE * IMG_SIZE * 3)
        extractor = RawFeatures(dest_shape, src_shape)
        features = extractor.extract(data)
        assert(features.shape == ((IMG_SIZE / 2) ** 2, ))

    def test_extract_several(self):
        src_shape = (IMG_SIZE, IMG_SIZE)
        dest_shape = (IMG_SIZE / 2, IMG_SIZE / 2)
        data = np.random.rand(4, IMG_SIZE * IMG_SIZE)
        extractor = RawFeatures(dest_shape, src_shape)
        features = extractor.extract(data)
        assert(features.ndim == 2)
        assert(features.shape[0] == data.shape[0])
        assert(features.shape[1] == (IMG_SIZE / 2) ** 2)
