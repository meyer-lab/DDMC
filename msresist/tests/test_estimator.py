"""
Testing file for a combined KMeans/PLSR estimator.
"""
import unittest
import numpy as np
from sklearn.datasets import load_diabetes
from ..estimator import kmeansPLSR


class TestEstimator(unittest.TestCase):
    """ Testing class for a combined KMeans/PLSR estimator. """

    def test_fitting(self):
        """ Check that we can in fact perform fitting in one case. """

        X, y = load_diabetes(return_X_y=True)

        est = kmeansPLSR(3, 2)

        est.fit(X, y)
