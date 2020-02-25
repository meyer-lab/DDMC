"""
Testing file for the clustering methods by data and sequence.
"""

import unittest
import numpy as np
from ..clustering import MassSpecClustering
from ..sequence_analysis import preprocess_seqs, gmm_initialize
from ..pre_processing import preprocessing


class TestClustering(unittest.TestCase):
    """ Testing class for a clustering estimator. """

    def test_clusters(self):
        # """ Test that EMclustering is working by comparing with GMM clusters. """
        ABC = preprocessing(AXLwt=True, motifs=True, Vfilter=True, FCfilter=True, log2T=True)
        ABC = preprocess_seqs(ABC, "Y")
        data = ABC.iloc[:, 7:].T
        info = ABC.iloc[:, :7]
        ncl = 2
        GMMweight = 0.75
        pYTS = "Y"
        distance_method = "PAM250"
        covariance_type = "diag"
        max_n_iter = 20
        fooCV = np.arange(10)

        MSC = MassSpecClustering(info, ncl, GMMweight=GMMweight, distance_method=distance_method).fit(data, fooCV)
        Cl_seqs = MSC.cl_seqs_

        _, gmm_cl, _ = gmm_initialize(ABC, ncl, covariance_type, distance_method, pYTS)
        gmm_cl = [[str(seq) for seq in cluster] for cluster in gmm_cl]

        # assert that EM clusters are different than GMM clusters
        self.assertNotEqual(Cl_seqs, gmm_cl)
