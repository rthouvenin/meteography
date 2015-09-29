# -*- coding: utf-8 -*-

import random

import numpy as np
import pytest

from meteography.dataset import DataSet
from meteography.dataset import ImageSet


def make_filedict(t, jitter=False):
    return {
        'shape': (20, 20),
        'time': t + (random.randint(-2, 2) if jitter else 0),
        'data': np.random.rand(20*20)
    }


@pytest.fixture(scope='class')
def imageset():
    images = [make_filedict(t) for t in [4200, 420, 4242, 42]]
    return ImageSet(images)


@pytest.fixture
def bigimageset():
    random.seed(42)  # Reproducible tests
    images = [make_filedict(t, True) for t in range(6000, 12000, 60)]
    return ImageSet(images)


class TestImageSet:
    def helper_closest_match(self, imageset, start, interval, expected):
        closest = imageset.find_closest(start, interval)
        assert(closest is not None)
        assert(closest['time'] == expected)

    def helper_closest_nomatch(self, imageset, start, interval):
        closest = imageset.find_closest(start, interval)
        assert(closest is None)

    def test_init(self, imageset):
        "The first element should have the smallest time"
        assert(imageset.time(0) == 42)

    def test_closest_last_inrange(self, imageset):
        "The target is past the last element, the last element is acceptable"
        self.helper_closest_match(imageset, 4210, 50, 4242)

    def test_closest_last_outrange(self, imageset):
        "The target is past the last element, the last element is not acceptable"
        self.helper_closest_nomatch(imageset, 4250, 50)

    def test_closest_exactmatch(self, imageset):
        "The target exists in the set"
        self.helper_closest_match(imageset, 400, 20, 420)

    def test_closest_matchleft_inrange(self, imageset):
        "The closest match is lower than the target and acceptable"
        self.helper_closest_match(imageset, 400, 40, 420)

    def test_closest_matchleft_outrange(self, imageset):
        "The closest match is lower than the target but not acceptable"
        self.helper_closest_nomatch(imageset, 420, 40)

    def test_closest_matchright_inrange(self, imageset):
        "The closest match is greater than the target and acceptable"
        self.helper_closest_match(imageset, 4100, 150, 4242)

    def test_closest_matchright_outrange(self, imageset):
        "The closest match is greater than the target but not acceptable"
        self.helper_closest_nomatch(imageset, 4000, 50)

    def test_reduce_fewimg(self, bigimageset):
        "More pixels than images"
        bigimageset.reduce_dim()
        img0 = bigimageset[0]
        assert(len(img0['data']) <= len(bigimageset))


class TestDataSet:
    #Using a class scope, we also check that calling split multiple times
    #is not affected by previous splits
    @staticmethod
    @pytest.fixture
    def dataset(bigimageset):
        return DataSet.make(bigimageset, 5, 60, 120)

    @staticmethod
    @pytest.fixture
    def freshdataset(bigimageset):
        return DataSet.make(bigimageset, 5, 60, 120)

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

    def test_make_reducedset(self, bigimageset):
        bigimageset.reduce_dim()
        dataset = DataSet.make(bigimageset, 5, 60, 120)
        assert(dataset.output_data.shape[1] == len(bigimageset[0]['data']))

    def test_finddatapoints(self, bigimageset):
        "The list of data points should have the expected dimensions"
        interval, pred_time = 60, 120
        data_points = DataSet._find_datapoints(bigimageset, 5, interval,
                                               pred_time)
        # (max - future_time - (hist_len - 1)*interval - min) / step + 1
        # = (2000 - 20 - 4*10 - 1000) / 10 + 1 = 95
        assert(len(data_points) == 95)
        X, y = random.choice(data_points)
        assert(len(X) == 5)
        intervals = [X[i+1]['time'] - X[i]['time'] for i in range(len(X)-1)]
        assert(min(intervals) <= 2*interval)
        assert(y['time'] - X[-1]['time'] <= 2*pred_time)

    def test_split_60_30(self, dataset):
        "default = 70% (66.5) training, 15% (14.25) validation, 15% test"
        dataset.split()
        self.check_length(dataset, 66, 14, 15)

    def test_split_80_20(self, dataset):
        "80% (76) training, 20% (19) validation, 0 test"
        assert(len(dataset.input_data) == 95)
        dataset.split(.8, .2)
        self.check_length(dataset, 76, 19, 0)
