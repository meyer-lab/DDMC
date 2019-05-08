"""
Testing file for the chained methods.
"""
import unittest
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.cluster import KMeans
from ..plsr import ClusterAverages


class TestEstimator(unittest.TestCase):
    """ Testing class for a chained KMeans/PLSR estimator. """

    def test_fitting(self):
        """ Test that ClusterAverages is working by comparing to cluster centers. """

        X = load_diabetes(return_X_y=False)['data']

        n_clusters = 3

        kmeans = KMeans(init="k-means++", n_clusters=n_clusters)

        cluster_assignments = kmeans.fit_predict(X.T) 
        X_Filt_Clust_Avgs = ClusterAverages(X, cluster_assignments, n_clusters, X.shape[0])

        # Assert that the cluster centers are also their averages from assignments
        self.assertTrue(np.allclose(np.array(kmeans.cluster_centers_).T, X_Filt_Clust_Avgs))
