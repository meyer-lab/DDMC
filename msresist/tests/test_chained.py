"""
Testing file for the chained methods.
"""
import unittest
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.cluster import KMeans
from ..plsr import ClusterAverages
import random, string


class TestEstimator(unittest.TestCase):
    """ Testing class for a chained KMeans/PLSR estimator. """

    def test_fitting(self):
        """ Test that ClusterAverages is working by comparing to cluster centers. """

        X = load_diabetes(return_X_y=False)['data']
        
        MadeUp_ProtNames, MadeUp_peptide_phosphosite = [], []
        for i in range(X.shape[0]):
            MadeUp_ProtNames.append(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)))
            MadeUp_peptide_phosphosite.append(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)))
        n_clusters = 3

        kmeans = KMeans(init="k-means++", n_clusters=n_clusters)

        cluster_assignments = kmeans.fit_predict(X.T) 
        centers, DictClusterToMembers = ClusterAverages(X, cluster_assignments, n_clusters, X.shape[0], MadeUp_ProtNames, MadeUp_peptide_phosphosite)

        # Assert that the cluster centers are also their averages from assignments
        self.assertTrue(np.allclose(np.array(kmeans.cluster_centers_).T, centers))
