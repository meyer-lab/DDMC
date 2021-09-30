from pomegranate import *
from pomegranate.io import DataGenerator

from nose.tools import with_setup
from nose.tools import assert_true
from nose.tools import assert_equal
from nose.tools import assert_raises
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

import random
import numpy
import pandas

numpy.random.seed(0)
random.seed(0)

nan = numpy.nan


def test_io_datagenerator_shape():
    X = numpy.random.randn(500, 13)
    data = DataGenerator(X)

    assert_array_equal(data.shape, X.shape)


def test_io_datagenerator_classes_fail():
    X = numpy.random.randn(500, 13)
    data = DataGenerator(X)

    assert_raises(ValueError, lambda data: data.classes, data)


def test_io_datagenerator_classes():
    X = numpy.random.randn(500, 13)
    y = numpy.random.randint(5, size=500)
    data = DataGenerator(X, y=y)

    assert_array_equal(data.classes, [0, 1, 2, 3, 4])


def test_io_datagenerator_x_batches():
    X = numpy.random.randn(500, 13)
    w = numpy.ones(500)

    data = DataGenerator(X)
    X_ = numpy.concatenate([batch[0] for batch in data.batches()])
    w_ = numpy.concatenate([batch[1] for batch in data.batches()])
    assert_almost_equal(X, X_)
    assert_almost_equal(w, w_)

    data = DataGenerator(X, batch_size=123)
    X_ = numpy.concatenate([batch[0] for batch in data.batches()])
    w_ = numpy.concatenate([batch[1] for batch in data.batches()])
    assert_almost_equal(X, X_)
    assert_almost_equal(w, w_)

    data = DataGenerator(X, batch_size=1)
    X_ = numpy.concatenate([batch[0] for batch in data.batches()])
    w_ = numpy.concatenate([batch[1] for batch in data.batches()])
    assert_almost_equal(X, X_)
    assert_almost_equal(w, w_)

    data = DataGenerator(X, batch_size=506)
    X_ = numpy.concatenate([batch[0] for batch in data.batches()])
    w_ = numpy.concatenate([batch[1] for batch in data.batches()])
    assert_almost_equal(X, X_)
    assert_almost_equal(w, w_)


def test_io_datagenerator_w_batches():
    X = numpy.random.randn(500, 13)
    w = numpy.abs(numpy.random.randn(500))

    data = DataGenerator(X, w)
    w_ = numpy.concatenate([batch[1] for batch in data.batches()])
    assert_almost_equal(w, w_)

    data = DataGenerator(X, w, batch_size=123)
    w_ = numpy.concatenate([batch[1] for batch in data.batches()])
    assert_almost_equal(w, w_)

    data = DataGenerator(X, w, batch_size=1)
    w_ = numpy.concatenate([batch[1] for batch in data.batches()])
    assert_almost_equal(w, w_)

    data = DataGenerator(X, w, batch_size=506)
    w_ = numpy.concatenate([batch[1] for batch in data.batches()])
    assert_almost_equal(w, w_)


def test_io_datagenerator_y_batches():
    X = numpy.random.randn(500, 13)
    w = numpy.abs(numpy.random.randn(500))
    y = numpy.random.randint(5, size=500)

    data = DataGenerator(X, y=y)
    y_ = numpy.concatenate([batch[1] for batch in data.batches()])
    assert_almost_equal(y, y_)

    data = DataGenerator(X, y=y, batch_size=123)
    y_ = numpy.concatenate([batch[1] for batch in data.batches()])
    assert_almost_equal(y, y_)

    data = DataGenerator(X, y=y, batch_size=1)
    y_ = numpy.concatenate([batch[1] for batch in data.batches()])
    assert_almost_equal(y, y_)

    data = DataGenerator(X, y=y, batch_size=506)
    y_ = numpy.concatenate([batch[1] for batch in data.batches()])
    assert_almost_equal(y, y_)


def test_io_datagenerator_wy_batches():
    X = numpy.random.randn(500, 13)
    w = numpy.abs(numpy.random.randn(500))
    y = numpy.random.randint(5, size=500)

    data = DataGenerator(X, w, y)
    y_ = numpy.concatenate([batch[1] for batch in data.batches()])
    w_ = numpy.concatenate([batch[2] for batch in data.batches()])
    assert_almost_equal(y, y_)
    assert_almost_equal(w, w_)

    data = DataGenerator(X, w, y, batch_size=123)
    y_ = numpy.concatenate([batch[1] for batch in data.batches()])
    w_ = numpy.concatenate([batch[2] for batch in data.batches()])
    assert_almost_equal(y, y_)
    assert_almost_equal(w, w_)

    data = DataGenerator(X, w, y, batch_size=1)
    y_ = numpy.concatenate([batch[1] for batch in data.batches()])
    w_ = numpy.concatenate([batch[2] for batch in data.batches()])
    assert_almost_equal(y, y_)
    assert_almost_equal(w, w_)

    data = DataGenerator(X, w, y, batch_size=506)
    y_ = numpy.concatenate([batch[1] for batch in data.batches()])
    w_ = numpy.concatenate([batch[2] for batch in data.batches()])
    assert_almost_equal(y, y_)
    assert_almost_equal(w, w_)


def test_io_datagenerator_wy_unlabeled():
    X = numpy.random.randn(500, 13)
    w = numpy.abs(numpy.random.randn(500))
    y = numpy.random.randint(5, size=500) - 1

    data = DataGenerator(X, w, y)
    X_ = numpy.concatenate([batch[0] for batch in data.unlabeled_batches()])
    w_ = numpy.concatenate([batch[1] for batch in data.unlabeled_batches()])

    assert_true(X.shape[0] > X_.shape[0])
    assert_almost_equal(X[y == -1], X_)
    assert_almost_equal(w[y == -1], w_)


def test_io_datagenerator_wy_labeled():
    X = numpy.random.randn(500, 13)
    w = numpy.abs(numpy.random.randn(500))
    y = numpy.random.randint(5, size=500) - 1

    data = DataGenerator(X, w, y)
    X_ = numpy.concatenate([batch[0] for batch in data.labeled_batches()])
    y_ = numpy.concatenate([batch[1] for batch in data.labeled_batches()])
    w_ = numpy.concatenate([batch[2] for batch in data.labeled_batches()])

    assert_true(X.shape[0] > X_.shape[0])
    assert_almost_equal(X[y != -1], X_)
    assert_almost_equal(y[y != -1], y_)
    assert_almost_equal(w[y != -1], w_)
