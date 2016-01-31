# -*- coding: utf-8 -*-

import random

import numpy as np
from PIL import Image
import pytest

from meteography.dataset import DataSet
from meteography.dataset import ImageSet

IMG_SIZE = 20


def make_filedict(t, jitter=False):
    return {
        'name': str(t) + '.png',
        'img_time': t + (random.randint(-2, 2) if jitter else 0),
        'pixels': np.random.rand(IMG_SIZE*IMG_SIZE)
    }


def imgset_fixture(request, tmpdir_factory, fname, images):
    filename = tmpdir_factory.mktemp('h5').join(fname).strpath
    imageset = ImageSet.create(filename, (IMG_SIZE, IMG_SIZE))
    for img in images:
        imageset._add_image(**img)
    imageset.fileh.flush()

    def fin():
        imageset.close()
    request.addfinalizer(fin)
    return imageset


@pytest.fixture(scope='session')
def imgfile(tmpdir_factory, t=12000):
    filename = tmpdir_factory.mktemp('img').join('%d.jpg' % t).strpath
    pixels = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE))
    img = Image.fromarray(pixels, mode='L')
    img.save(filename)
    return filename

@pytest.fixture(scope='session')
def times():
    return range(6000, 12000, 60)

@pytest.fixture(scope='session')
def filedimages(tmpdir_factory, times, jitter=True):
    random.seed(42)  # Reproducible tests
    images = [make_filedict(t, jitter) for t in times]
    for img in images:
        img['name'] = imgfile(tmpdir_factory, img['img_time'])
    return images

@pytest.fixture(scope='class')
def imageset(request, tmpdir_factory):
    images = filedimages(tmpdir_factory, [4200, 420, 4242, 42], False)
    return imgset_fixture(request, tmpdir_factory, 'imageset.h5', images)

@pytest.fixture(scope='session')
def bigimageset(request, tmpdir_factory, filedimages):
    return imgset_fixture(request, tmpdir_factory, 'bigimageset.h5', filedimages)

@pytest.fixture
def bigimageset_rw(request, tmpdir_factory, filedimages):
    "For tests that will modify the set"
    return bigimageset(request, tmpdir_factory, filedimages)

@pytest.fixture
def emptyimageset(request, tmpdir_factory):
    images = []
    return imgset_fixture(request, tmpdir_factory, 'emptyset.h5', images)


class TestImageSet:
    def helper_closest_match(self, imageset, start, interval, expected):
        closest = imageset.find_closest(start, interval)
        assert(closest is not None)
        assert(closest['time'] == expected)

    def helper_closest_nomatch(self, imageset, start, interval):
        closest = imageset.find_closest(start, interval)
        assert(closest is None)

    def test_pixelsat_int(self, imageset):
        "get_image and get_pixels_at should produce same pixels"
        img1 = imageset.get_image(42)
        pix2 = imageset.get_pixels_at(img1['id'])
        assert(np.allclose(img1['pixels'], pix2))

    def test_pixelsat_seq(self, imageset):
        "Read pixels from a tuple of indices"
        img1 = imageset.get_image(42)
        pixels = imageset.get_pixels_at((img1['id'], 0))
        assert(np.allclose(img1['pixels'], pixels[0]))
        assert(not np.allclose(pixels[1], pixels[0]))

    def test_pixelsat_seq_reduced(self, imageset):
        "Read reduced pixels from a tuple of indices"
        imageset.reduce_dim()
        self.test_pixelsat_seq(imageset)

    def test_addfromfile(self, imageset, imgfile):
        "Adding a file should increase the length of the set"
        prev_len = len(imageset)
        img = imageset.add_from_file(imgfile)
        assert(len(imageset) == prev_len+1)
        assert(img['id'] == prev_len)

    def test_addfromfile_twice(self, imageset, imgfile):
        "Adding an existing timestamp should not increase the length"
        img = imageset.add_from_file(imgfile)
        prev_len = len(imageset)
        img = imageset.add_from_file(imgfile)
        assert(len(imageset) == prev_len)
        assert(img['id'] == prev_len-1)

    def test_addfromfile_reduced(self, bigimageset_rw, imgfile):
        "Adding a file should still work after reducing the set"
        bigimageset_rw.reduce_dim()
        prev_len = len(bigimageset_rw)
        bigimageset_rw.add_from_file(imgfile)
        assert(len(bigimageset_rw) == prev_len+1)

    def test_closest_last_inrange(self, imageset):
        "The target is past the last element, the last element is acceptable"
        self.helper_closest_match(imageset, 4280, 50, 4242)

    def test_closest_last_outrange(self, imageset):
        "The target is past the last element, the last element is not acceptable"
        self.helper_closest_nomatch(imageset, 4350, 50)

    def test_closest_exactmatch(self, imageset):
        "The target exists in the set"
        self.helper_closest_match(imageset, 440, 20, 420)

    def test_closest_matchleft_inrange(self, imageset):
        "The closest match is lower than the target and acceptable"
        self.helper_closest_match(imageset, 440, 40, 420)

    def test_closest_matchleft_outrange(self, imageset):
        "The closest match is lower than the target but not acceptable"
        self.helper_closest_nomatch(imageset, 420, 40)

    def test_closest_matchright_inrange(self, imageset):
        "The closest match is greater than the target and acceptable"
        self.helper_closest_match(imageset, 4380, 150, 4242)

    def test_closest_matchright_outrange(self, imageset):
        "The closest match is greater than the target but not acceptable"
        self.helper_closest_nomatch(imageset, 4000, 50)

    def test_reduce_fewimg(self, bigimageset_rw):
        "More pixels than images should lead to less components than images"
        prev_length = len(bigimageset_rw)
        bigimageset_rw.reduce_dim()
        img0 = bigimageset_rw.get_pixels_at(0, 'pca')[0]  # FIXME pca is hardcoded
        assert(len(bigimageset_rw) == prev_length)
        assert('pca' in bigimageset_rw.feature_sets)
        assert(len(img0) <= len(bigimageset_rw) < IMG_SIZE*IMG_SIZE)

    def test_reduce_twice(self, bigimageset_rw):
        "Reducing twice the same set should not cause any problem"
        prev_length = len(bigimageset_rw)
        bigimageset_rw.reduce_dim()
        bigimageset_rw.reduce_dim()
        img0 = bigimageset_rw.get_pixels_at(0, 'pca')[0]  # FIXME pca is hardcoded
        assert(len(bigimageset_rw) == prev_length)
        assert('pca' in bigimageset_rw.feature_sets)
        assert(len(img0) <= len(bigimageset_rw) < IMG_SIZE*IMG_SIZE)

    def test_reduce_smallsample(self, bigimageset_rw):
        "More images than sample size"
        prev_length = len(bigimageset_rw)
        sample_size = prev_length / 2
        bigimageset_rw.reduce_dim(sample_size, None)
        img0 = bigimageset_rw.get_pixels_at(0, 'pca')[0]  # FIXME pca is hardcoded
        assert(len(bigimageset_rw) == prev_length)
        assert('pca' in bigimageset_rw.feature_sets)
        assert(len(img0) == sample_size)

    def test_reduce_manyimg(self, bigimageset_rw, tmpdir_factory):
        """More images than pixels but less than sample size
        won't reduce the dimensionality, even after a first reduction"""
        bigimageset_rw.reduce_dim()
        images = [make_filedict(t, True) for t in range(6000, 30000, 60)]
        for img in images:
            img['name'] = imgfile(tmpdir_factory, img['img_time'])
            bigimageset_rw._add_image(**img)
        bigimageset_rw.fileh.flush()
        prev_length = len(bigimageset_rw)
        bigimageset_rw.reduce_dim(450, None)
        img0 = bigimageset_rw.get_pixels_at(0, 'pca')[0]  # FIXME pca is hardcoded
        assert(len(bigimageset_rw) == prev_length)
        assert(len(img0) == IMG_SIZE*IMG_SIZE)


@pytest.fixture(scope='class')
def dataset(bigimageset):
    dataset = DataSet.create(bigimageset)
    intervals = [60] * 4 + [120]
    dataset.make_set('test', intervals)
    return dataset

@pytest.fixture
def dataset_rw(bigimageset_rw):
    "For tests that will modify the dataset"
    return dataset(bigimageset_rw)

@pytest.fixture
def emptydataset(emptyimageset):
    dataset = DataSet.create(emptyimageset)
    return dataset


class TestDataSet:
    def check_length(self, ds, train, valid, test):
        assert(len(ds.train_input) == train)
        assert(len(ds.train_output) == train)
        assert(len(ds.valid_input) == valid)
        assert(len(ds.valid_output) == valid)
        assert(len(ds.test_input) == test)
        assert(len(ds.test_output) == test)

    def check_width(self, dataset, max_size):
        assert(dataset.input_data.shape[1] <= (max_size+1)*dataset.history_len)
        assert(dataset.output_data.shape[1] <= max_size)

    def test_findexamples(self, dataset):
        "The list of data points should have the expected dimensions"
        interval, pred_time = 60, 120
        intervals = [60] * 4 + [pred_time]
        feature_set = dataset.imgset.feature_sets.keys()[0]
        data_points = list(dataset._find_examples(intervals, feature_set))
        # (max - future_time - (hist_len - 1)*interval - min) / step + 1
        # = (12000 - 120 - 4*60 - 6000) / 60 = 94
        assert(len(data_points) == 94)
        X, y = random.choice(data_points)
        assert(len(X) == 5)
        intervals = [X[i+1]['time'] - X[i]['time'] for i in range(len(X)-1)]
        assert(min(intervals) <= 2*interval)
        assert(y['time'] - X[-1]['time'] <= 2*pred_time)

    def test_findexamples_empty(self, emptydataset):
        "An empty imageset should not be a problem"
        feature_set = emptydataset.imgset.feature_sets.keys()[0]
        data_points = list(emptydataset._find_examples([6, 6, 6], feature_set))
        assert(len(data_points) == 0)

    def test_makeset_empty(self, emptydataset):
        "An empty imageset should not be a problem"
        intervals = [60] * 2 + [1800]
        newset = emptydataset.make_set('empty', intervals)
        assert(len(newset.input) == 0)

    def test_makeinput_last(self, dataset):
        """Making an input that does not exist in the set of examples, with a
        target time different from the set"""
        #11942 is the last image of bigimageset
        row = dataset.make_input('test', 11942, 666)
        assert(row is not None)
        assert(len(row) == 5 * (IMG_SIZE*IMG_SIZE+1))
        assert(row[-1] == 666)

    def test_add_toempty(self, emptydataset, imgfile):
        """Adding an image to an empty dataset should add an image but cannot
        create an example"""
        testset = emptydataset.init_set('test', intervals=[6, 6, 6])
        emptydataset.add_image(imgfile, 'test')
        assert(len(emptydataset.imgset) == 1)
        assert(len(testset.input) == 0)
        assert(len(testset.output) == 0)

    def test_add_newexample(self, dataset_rw, imgfile):
        "Adding an image to a compatible dataset should create a new example"
        prev_length = len(dataset_rw.imgset)
        testset = dataset_rw.fileh.root.examples.test
        prev_nb_ex = len(testset.input)
        dataset_rw.add_image(imgfile, 'test')
        assert(len(dataset_rw.imgset) == prev_length + 1)
        assert(len(testset.input) == prev_nb_ex + 1)
        assert(len(testset.output) == prev_nb_ex + 1)

    def test_reducedim(self, dataset_rw):
        testset = dataset_rw.fileh.root.examples.test
        prev_refs_shape = testset.img_refs.shape
        prev_nb_feat = testset.input.shape[1]
        prev_imgdim = testset.output.shape[1]
        dataset_rw.reduce_dim()
        assert(testset.img_refs.shape == prev_refs_shape)
        assert(len(testset.input) == prev_refs_shape[0])
        assert(len(testset.output) == prev_refs_shape[0])
        assert(testset.input.shape[1] < prev_nb_feat)
        assert(testset.output.shape[1] < prev_imgdim)

    def test_add_afterreduced(self, dataset_rw, imgfile):
        """Adding an image to a dataset after it was reduced should
        not be a problem"""
        prev_length = len(dataset_rw.imgset)
        testset = dataset_rw.fileh.root.examples.test
        prev_nb_ex = len(testset.input)
        dataset_rw.reduce_dim()
        dataset_rw.add_image(imgfile, 'test')
        assert(len(dataset_rw.imgset) == prev_length + 1)
        assert(len(testset.input) == prev_nb_ex + 1)
        assert(len(testset.output) == prev_nb_ex + 1)

    def test_repack(self, dataset_rw):
        "Repacking should reclaim the space of a deleted set"
        prev_size = dataset_rw.fileh.get_filesize()
        dataset_rw.delete_set('test')
        dataset_rw.repack()
        new_size = dataset_rw.fileh.get_filesize()
        assert(new_size < prev_size)

    def test_getoutput(self, dataset):
        pixels = dataset.output_img('test', 0)
        assert(pixels.shape == (20, 20))
