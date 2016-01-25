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
def filedimageset(request, tmpdir_factory):
    "Same as bigimageset, but image files actually exist"
    random.seed(42)  # Reproducible tests
    images = [make_filedict(t, True) for t in range(6000, 12000, 60)]
    for img in images:
        img['name'] = imgfile(tmpdir_factory, img['img_time'])
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

    def test_addfromfile_reduced(self, bigimageset, imgfile):
        "Adding a file should still work after reducing the set"
        bigimageset.reduce_dim()
        prev_len = len(bigimageset)
        bigimageset.add_from_file(imgfile)
        assert(len(bigimageset) == prev_len+1)

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
        prev_length = len(bigimageset)
        bigimageset.reduce_dim()
        img0 = next(iter(bigimageset))
        assert(len(bigimageset) == prev_length)
        assert(bigimageset.pca is not None)
        assert(bigimageset.fileh.root.images.pcamodel is not None)
        assert(len(img0['pixels']) <= len(bigimageset) < IMG_SIZE*IMG_SIZE)

    def test_reduce_twice(self, filedimageset):
        "Reducing twice the same set should not cause any problem"
        prev_length = len(filedimageset)
        filedimageset.reduce_dim()
        filedimageset.reduce_dim()
        img0 = next(iter(filedimageset))
        assert(len(filedimageset) == prev_length)
        assert(filedimageset.pca is not None)
        assert(filedimageset.fileh.root.images.pcamodel is not None)
        assert(len(img0['pixels']) <= len(filedimageset) < IMG_SIZE*IMG_SIZE)

    def test_reduce_smallsample(self, bigimageset):
        "More images than sample size"
        prev_length = len(bigimageset)
        sample_size = prev_length / 2
        bigimageset.reduce_dim(sample_size, None)
        img0 = next(iter(bigimageset))
        assert(len(bigimageset) == prev_length)
        assert(bigimageset.pca is not None)
        assert(bigimageset.fileh.root.images.pcamodel is not None)
        assert(len(img0['pixels']) == sample_size)

    def test_reduce_manyimg(self, filedimageset, tmpdir_factory):
        """More images than pixels but less than sample size
        won't reduce the dimensionality, even after a first reduction"""
        filedimageset.reduce_dim()
        images = [make_filedict(t, True) for t in range(6000, 30000, 60)]
        for img in images:
            img['name'] = imgfile(tmpdir_factory, img['img_time'])
            filedimageset._add_image(**img)
        filedimageset.fileh.flush()
        prev_length = len(filedimageset)
        filedimageset.reduce_dim(450, None)
        img0 = next(iter(filedimageset))
        assert(len(filedimageset) == prev_length)
        assert(len(img0['pixels']) == IMG_SIZE*IMG_SIZE)


@pytest.fixture
def dataset(bigimageset):
    dataset = DataSet.create(bigimageset)
    dataset.make_set('test', 5, 60, 120)
    return dataset

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
        data_points = list(dataset._find_examples(intervals))
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
        data_points = list(emptydataset._find_examples([6, 6, 6]))
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

    def test_add_newexample(self, dataset, imgfile):
        "Adding an image to a compatible dataset should create a new example"
        prev_length = len(dataset.imgset)
        testset = dataset.fileh.root.examples.test
        prev_nb_ex = len(testset.input)
        dataset.add_image(imgfile, 'test')
        assert(len(dataset.imgset) == prev_length + 1)
        assert(len(testset.input) == prev_nb_ex + 1)
        assert(len(testset.output) == prev_nb_ex + 1)

    def test_reducedim(self, dataset):
        testset = dataset.fileh.root.examples.test
        prev_refs_shape = testset.img_refs.shape
        prev_nb_feat = testset.input.shape[1]
        prev_imgdim = testset.output.shape[1]
        dataset.reduce_dim()
        assert(testset.img_refs.shape == prev_refs_shape)
        assert(len(testset.input) == prev_refs_shape[0])
        assert(len(testset.output) == prev_refs_shape[0])
        assert(testset.input.shape[1] < prev_nb_feat)
        assert(testset.output.shape[1] < prev_imgdim)

    def test_add_afterreduced(self, dataset, imgfile):
        """Adding an image to a dataset after it was reduced should
        not be a problem"""
        prev_length = len(dataset.imgset)
        testset = dataset.fileh.root.examples.test
        prev_nb_ex = len(testset.input)
        dataset.reduce_dim()
        dataset.add_image(imgfile, 'test')
        assert(len(dataset.imgset) == prev_length + 1)
        assert(len(testset.input) == prev_nb_ex + 1)
        assert(len(testset.output) == prev_nb_ex + 1)

    def test_repack(self, dataset):
        "Repacking should reclaim the space of a deleted set"
        prev_size = dataset.fileh.get_filesize()
        dataset.delete_set('test')
        dataset.repack()
        new_size = dataset.fileh.get_filesize()
        assert(new_size < prev_size)

    def test_getoutput(self, dataset):
        pixels = dataset.output_img('test', 0)
        assert(pixels.shape == (20, 20))
