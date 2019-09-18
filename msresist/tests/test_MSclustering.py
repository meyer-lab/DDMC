"""
Testing file for the clustering methods by data and sequence.
"""
import unittest
from ..sequence_analysis import EMclustering, preprocess_seqs, gmm_init_clusters, gmm_pvalue
from ..pre_processing import preprocessing

class TestClustering(unittest.TestCase):
    """ Testing class for a clustering estimator. """

    def test_clusters(self):
        """ Test that EMclustering is working by comparing with GMM clusters. """
        ABC = preprocessing(motifs=True, Vfilter=True, FCfilter=True, log2T=True)
        ABC = preprocess_seqs(ABC, "Y")
        Cl_seqs, _, _, _ = EMclustering(ABC, 4, 1, "Y", "tied", 20)
        X, _ = gmm_pvalue(ABC, 4, "tied")
        gmm_cl = gmm_init_clusters(X, "Y", 4)
        gmm_cl = [[str(seq) for seq in cluster] for cluster in gmm_cl]

        #assert that EM clusters are different than GMM clusters
        self.assertNotEqual(Cl_seqs, gmm_cl)
