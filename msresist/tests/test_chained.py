"""
Testing file for the chained methods.
"""
import unittest
import random
import string
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.cluster import KMeans
from ..clustering import ClusterAverages

class TestEstimator(unittest.TestCase):
    """ Testing class for a chained KMeans/PLSR estimator. """

    def test_fitting(self):
        """ Test that ClusterAverages is working by comparing to cluster centers. """
        X = pd.DataFrame(load_diabetes(return_X_y=False)['data'])
        kmeans = KMeans(init="k-means++", n_clusters=4)
        cluster_assignments = kmeans.fit_predict(X.T)
        centers, DictClusterToMembers = ClusterAverages(X, cluster_assignments)

        # Assert that the cluster centers are also their averages from assignments
        self.assertTrue(np.allclose(np.array(kmeans.cluster_centers_).T, centers))
