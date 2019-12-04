"""
Testing file for the clustering methods by data and sequence.
"""
import unittest
from ..sequence_analysis import EM_clustering, preprocess_seqs, gmm_initialCl_and_pvalues
from ..pre_processing import preprocessing


class TestClustering(unittest.TestCase):
    """ Testing class for a clustering estimator. """

    def test_clusters(self):
        """ Test that EMclustering is working by comparing with GMM clusters. """
        ABC = preprocessing(motifs=True, Vfilter=True, FCfilter=True, log2T=True)
        ABC = preprocess_seqs(ABC, "Y")
        data = ABC.iloc[:, 2:12].T
        seqs = ABC.iloc[:, 0]
        names = ABC.iloc[:, 1]
        ncl = 4
        GMMweight = 1
        pYTS = "Y"
        distance_method = "PAM250"
        covariance_type = "diag"
        max_n_iter = 20

        Cl_seqs, _, _, _, _ = EM_clustering(data, seqs, names, ncl, GMMweight, pYTS, distance_method, covariance_type, max_n_iter)
        gmm_cl, _ = gmm_initialCl_and_pvalues(ABC, 4, "diag", "Y")
        gmm_cl = [[str(seq) for seq in cluster] for cluster in gmm_cl]

        # assert that EM clusters are different than GMM clusters
        self.assertNotEqual(Cl_seqs, gmm_cl)
