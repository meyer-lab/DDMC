"""
Testing file for the clustering methods by data and sequence.
"""

import pickle
import pytest
import numpy as np
from ..clustering import MassSpecClustering
from ..expectation_maximization import EM_clustering
from ..pre_processing import preprocessing


X = preprocessing(AXLwt_GF=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
data = X.select_dtypes(include=['float64']).T
info = X.select_dtypes(include=['object'])
preMotifSet = ["ABL", "EGFR", "ALK", "SRC", "YES"]


@pytest.mark.parametrize("distance_method", ["PAM250", "Binomial", "PAM250_fixed"])
def test_wins(distance_method):
    """ Test that EMclustering is working by comparing with GMM clusters. """
    MSC = MassSpecClustering(info, 2, SeqWeight=0, distance_method=distance_method, pre_motifs=preMotifSet[0:2]).fit(X=data)
    distances = MSC.wins(data)

    # assert that the distance to the same sequence weight is less
    assert distances[0] < 10.0
    assert distances[0] < distances[1]


@pytest.mark.parametrize("w", [0, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("ncl", [2, 5])
@pytest.mark.parametrize("distance_method", ["PAM250", "Binomial", "PAM250_fixed"])
def test_clusters(w, ncl, distance_method):
    """ Test that EMclustering is working by comparing with GMM clusters. """
    if (distance_method == "PAM250_fixed") and (ncl > 6):
        return

    MSC = MassSpecClustering(info, ncl, SeqWeight=w, distance_method=distance_method, pre_motifs=preMotifSet[0:ncl]).fit(X=data)

    # Assert that we got a reasonable result
    assert np.all(np.isfinite(MSC.scores_))
    assert np.all(np.isfinite(MSC.seq_scores_))


@pytest.mark.parametrize("distm", ["PAM250", "Binomial", "PAM250_fixed"])
def test_pickle(distm):
    """ Test that EMclustering can be pickled and unpickled. """
    MSC = MassSpecClustering(info, 3, SeqWeight=2, distance_method=distm, pre_motifs=preMotifSet[0:3]).fit(X=data)
    unpickled = pickle.loads(pickle.dumps(MSC))
    _, scores, _, _ = EM_clustering(data, info, 3, gmmIn=unpickled.gmm_)

    assert np.all(np.isfinite(unpickled.scores_))
    np.testing.assert_allclose(MSC.scores_, scores, rtol=0.5, atol=0.5)
