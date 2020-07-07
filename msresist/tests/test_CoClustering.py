"""
Testing file for the clustering methods by data and sequence.
"""

import unittest
import numpy as np
from ..clustering import MassSpecClustering
from ..gmm import gmm_initialize
from ..pre_processing import preprocessing


class TestClustering(unittest.TestCase):
    """ Testing class for a clustering estimator. """

    def test_clusters(self):
        # """ Test that EMclustering is working by comparing with GMM clusters. """
        X = preprocessing(AXLwt=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
        data = X.select_dtypes(include=['float64']).T
        info = X.select_dtypes(include=['object'])
        ncl = 2
        SeqWeights = [0, 3, 10000]
        distance_method = "Binomial"
        fooCV = np.arange(10)

        for w in SeqWeights:
            MSC = MassSpecClustering(info, ncl, SeqWeight=w, distance_method=distance_method).fit(data, fooCV)

        Cl_seqs = MSC.cl_seqs_

        _, gmm_cl, _, _ = gmm_initialize(X, ncl, distance_method)
        gmm_cl = [[str(seq) for seq in cluster] for cluster in gmm_cl]

        # assert that EM clusters are different than GMM clusters
        self.assertNotEqual(Cl_seqs, gmm_cl)
