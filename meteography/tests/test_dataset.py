# -*- coding: utf-8 -*-

import random

import numpy as np
from PIL import Image
import pytest

from meteography.dataset import DataSet
from meteography.dataset import ImageSet


def make_filedict(t, jitter=False):
    return {
        'name': str(t) + '.png',
        'img_time': t + (random.randint(-2, 2) if jitter else 0),
        'data': np.random.rand(20*20)
    }


def imgset_fixture(request, tmpdir_factory, fname, images):
    filename = tmpdir_factory.mktemp('h5').join(fname).strpath
    imageset = ImageSet.create(filename, (20, 20))
    for img in images:
        imageset._add_image(**img)
    imageset.fileh.flush()

    def fin():
        imageset.close()
    request.addfinalizer(fin)
    return imageset


@pytest.fixture(scope='session')
def imgfile(tmpdir_factory):
    filename = tmpdir_factory.mktemp('img').join('12000.jpg').strpath
    pixels = np.zeros((20, 20))
    img = Image.fromarray(pixels, mode='L')
    img.save(filename)
    return filename

@pytest.fixture(scope='class')
def imageset(request, tmpdir_factory):
    images = [make_filedict(t) for t in [4200, 420, 4242, 42]]
    return imgset_fixture(request, tmpdir_factory, 'imageset.h5', images)

@pytest.fixture
def bigimageset(request, tmpdir_factory):
    random.seed(42)  # Reproducible tests
    images = [make_filedict(t, True) for t in range(6000, 12000, 60)]
    return imgset_fixture(request, tmpdir_factory, 'bigimageset.h5', images)

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

    def test_sort(self, imageset):
        "The first element should have the smallest time"
        imageset.sort()
        img0 = next(iter(imageset))
        assert(img0['time'] == 42)

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

    def test_reduce_fewimg(self, bigimageset):
        "More pixels than images should lead to less components than images"
        bigimageset.reduce_dim()
        img0 = next(iter(bigimageset))
        assert(len(img0['data']) <= len(bigimageset))


@pytest.fixture
def dataset(bigimageset):
    dataset = DataSet.create(bigimageset.fileh, bigimageset)
    dataset.make_set('test', 5, 60, 120)
    return dataset

@pytest.fixture
def emptydataset(emptyimageset):
    dataset = DataSet.create(emptyimageset.fileh, emptyimageset)
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
        data_points = dataset._find_examples(intervals)
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
        data_points = emptydataset._find_examples([6, 6, 6])
        assert(len(data_points) == 0)

    def test_makeset_empty(self, emptydataset):
        "An empty imageset should not be a problem"
        newset = emptydataset.make_set('empty')
        assert(len(newset.input) == 0)

    def test_makeinput_last(self, dataset):
        """Making an input that does not exist in the set of examples, with a
        target time different from the set"""
        #11942 is the last image of bigimageset
        row = dataset.make_input('test', 11942, 666)
        assert(row is not None)
        assert(len(row) == 5 * (400+1))
        assert(row[-1] == 666)

    def test_add_toempty(self, emptydataset, imgfile):
        """Adding an image to an empty dataset should add an image but cannot
        create an example"""
        testset = emptydataset.init_set('test', intervals=[6, 6, 6])
        emptydataset.add_image('test', imgfile)
        assert(len(emptydataset.imgset) == 1)
        assert(len(testset.input) == 0)
        assert(len(testset.output) == 0)

    def test_add_newexample(self, dataset, imgfile):
        "Adding an image to a compatible dataset should create a new example"
        prev_length = len(dataset.imgset)
        prev_nb_ex = len(dataset.input_data)
        dataset.add_image('test', imgfile)
        assert(len(dataset.imgset) == prev_length + 1)
        assert(len(dataset.input_data) == prev_nb_ex + 1)
        assert(len(dataset.output_data) == prev_nb_ex + 1)

    def test_split_60_30(self, dataset):
        "default = 70% (66.5) training, 15% (14.25) validation, 15% test"
        dataset.split()
        self.check_length(dataset, 65, 14, 15)

    def test_split_80_20(self, dataset):
        "80% (76) training, 20% (19) validation, 0 test"
        assert(len(dataset.input_data) == 94)
        dataset.split(.8, .2)
        self.check_length(dataset, 75, 18, 1)
