import numpy


class BaseGenerator(object):
    """The base data generator class.

    This object is inherited by data generator objects in order to specify that
    they are data generators. Do not use this object directly.
    """

    def __init__(self):
        pass

    def __len__(self):
        return NotImplementedError

    @property
    def shape(self):
        return NotImplementedError

    @property
    def classes(self):
        return NotImplementedError

    @property
    def ndim(self):
        return NotImplementedError


class DataGenerator(BaseGenerator):
    """A generator that returns batches of a data set.

    This object will wrap a data set and optionally a set of labels and will
    return batches of data as requested. When it reaches the end of a data
    set it will not roll over but rather return a batch of data smaller
    than the other batches.

    Parameters
    ----------
    X : numpy.ndarray or list
            The data set to iterate over.

    weights : numpy.ndarray or list or None, optional
            The weights for each example. Default is None.

    y: numpy.ndarray or list or None, optional
            The set of labels for each example in the data set. Default is None.

    batch_size : int or None, optional
            The size of the batches to return. If None will return the full data
            set each time. Default is None

    batches_per_epoch : int or None, optional
            The number of batches to return before resetting the index. If the
            value is too low you may not see all examples from the data set. If
            None, will return enough batches to cover the entire data set. Default
            is None.
    """

    def __init__(self, X, weights=None, y=None, batch_size=None,
                 batches_per_epoch=None):
        self.X = numpy.array(X)
        self.y = y
        self.idx = 0

        if y is not None and len(y) != len(X):
            raise ValueError("Size of label vector y does not match size of data.")

        if weights is None:
            self.weights = numpy.ones(len(X), dtype='float64')
        else:
            if len(weights) != len(X):
                raise ValueError("Size of weight vector does not match size of data.")

            self.weights = numpy.array(weights)

        if batch_size is None:
            self.batch_size = len(self)
        else:
            self.batch_size = int(batch_size)

        if batches_per_epoch is None:
            self.batches_per_epoch = float("inf")
        else:
            self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        return len(self.X)

    @property
    def shape(self):
        return self.X.shape

    @property
    def classes(self):
        if self.y is None:
            raise ValueError("Classes cannot be found on an unlabeled data set.")

        return numpy.unique(self.y)

    @property
    def ndim(self):
        return self.X.ndim

    def batches(self):
        if self.batch_size == len(self):
            while True:
                if self.y is not None:
                    yield self.X, self.y, self.weights
                else:
                    yield self.X, self.weights
                break
        else:
            start, end = 0, self.batch_size
            iteration = 0

            while start < len(self) and iteration < self.batches_per_epoch:
                if self.y is not None:
                    yield (self.X[start:end], self.y[start:end],
                           self.weights[start:end])
                else:
                    yield self.X[start:end], self.weights[start:end]

                start += self.batch_size
                end += self.batch_size
                iteration += 1

    def labeled_batches(self):
        X = self.X[self.y != -1]
        weights = self.weights[self.y != -1]
        y = self.y[self.y != -1]

        start, end = 0, self.batch_size
        while start < len(X):
            yield X[start:end], y[start:end], weights[start:end]

            start += self.batch_size
            end += self.batch_size

    def unlabeled_batches(self):
        X = self.X[self.y == -1]
        weights = self.weights[self.y == -1]

        start, end = 0, self.batch_size
        while start < len(X):
            yield X[start:end], weights[start:end]

            start += self.batch_size
            end += self.batch_size
