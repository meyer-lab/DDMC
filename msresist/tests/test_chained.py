"""
Testing file for the chained methods.
"""
import unittest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from ..pre_processing import preprocessing
from ..clustering import ClusterAverages


class TestEstimator(unittest.TestCase):
    """ Testing class for a chained KMeans/PLSR estimator. """

    def test_fitting(self):
        """ Test that ClusterAverages is working by comparing to cluster centers. """
        X = preprocessing(Axlmuts_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
        d = X.select_dtypes(include=['float64']).T
        kmeans = KMeans(init="k-means++", n_clusters=4)
        cluster_assignments = kmeans.fit_predict(d.T)
        centers = ClusterAverages(d, cluster_assignments)

        # Assert that the cluster centers are also their averages from assignments
        self.assertTrue(np.allclose(np.array(kmeans.cluster_centers_).T, centers[0]))
