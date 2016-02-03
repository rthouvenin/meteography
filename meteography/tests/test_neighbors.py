# -*- coding: utf-8 -*-

import numpy as np
import pytest
import tables

from meteography.neighbors import NearestNeighbors


@pytest.fixture
def small_batch(request):
    backup = NearestNeighbors.BATCH_SIZE
    NearestNeighbors.BATCH_SIZE = 15
    def fin():
        NearestNeighbors.BATCH_SIZE = backup
    request.addfinalizer(fin)

@pytest.fixture
def fixed_array():
    nb_ex = 22
    nb_feat = 3
    X = np.empty((nb_ex, nb_feat))
    y = np.array(range(nb_ex))
    for i in y:
        X[i] = [i] * nb_feat
    return X, y

@pytest.fixture
def variable_array(request, tmpdir_factory):
    filename = tmpdir_factory.mktemp('h5').join('var.h5').strpath
    fp = tables.open_file(filename, mode='w')
    fp.create_earray(fp.root, 'X', shape=(0, 3), atom=tables.FloatAtom())
    fp.create_earray(fp.root, 'y', shape=(0,), atom=tables.FloatAtom())
    fp.flush()

    def fin():
        fp.close()
    request.addfinalizer(fin)
    return fp.root.X, fp.root.y


def test_init_batch(small_batch, fixed_array):
    "Check computation of batch"
    nn = NearestNeighbors()
    nn.fit(*fixed_array)
    assert(nn.batch_len == 5)  # (15 / 3)
    assert(nn.nb_batch == 5)  # 22 / 5 + 1
    assert(nn.batch.shape == (5+1, 3))


def test_predict_singlebatch(fixed_array):
    "An existing row should be found, only one batch"
    nn = NearestNeighbors()
    nn.fit(*fixed_array)
    y = nn.predict(np.array([17, 17, 17]))
    assert(y == 17)


def test_predict_multibatch(small_batch, fixed_array):
    "An existing row should be found, only one batch"
    nn = NearestNeighbors()
    nn.fit(*fixed_array)
    y = nn.predict(np.array([17, 17, 17]))
    assert(y == 17)


def test_predict_notexisting(fixed_array):
    "Find the nearest neighbor of an unexisting row"
    nn = NearestNeighbors()
    nn.fit(*fixed_array)
    y = nn.predict(np.array([4.6, 4.4, 4.4]))
    assert(y == 4)


def test_predict_wminkowski(fixed_array):
    "Weighing the metric should be taken into account"
    nn = NearestNeighbors(metric='wminkowski', w=np.array([10, 1, 1]))
    nn.fit(*fixed_array)
    y4 = nn.predict(np.array([4.4, 4.4, 4.6]))
    y5 = nn.predict(np.array([4.6, 4.4, 4.4]))
    assert(y4 == 4)
    assert(y5 == 5)


def test_predict_extendedarray(variable_array, fixed_array):
    "Extending the array after fitting it should not be a problem"
    X, y = variable_array
    nn = NearestNeighbors()
    nn.fit(X, y)
    newX, newy = fixed_array
    X.append(newX)
    y.append(newy)
    y = nn.predict(np.array([4.2, 4.2, 4.2]))
    assert(y == 4)
