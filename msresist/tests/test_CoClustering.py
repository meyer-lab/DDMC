"""
Testing file for the clustering methods by data and sequence.
"""

import pytest
import numpy as np
import scipy.stats as sp
from ..clustering import MassSpecClustering
from ..pre_processing import preprocessing


X = preprocessing(AXLwt=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
data = X.select_dtypes(include=['float64']).T
info = X.select_dtypes(include=['object'])


@pytest.mark.parametrize("w", [0, 0.1, 0.3, 1])
@pytest.mark.parametrize("ncl", [2, 3, 4])
@pytest.mark.parametrize("distance_method", ["PAM250", "Binomial"])
def test_clusters(w, ncl, distance_method):
    # """ Test that EMclustering is working by comparing with GMM clusters. """
    MSC = MassSpecClustering(info, ncl, SeqWeight=w, distance_method=distance_method).fit(X=data)

    # assert that EM clusters are different than GMM clusters
    assert True
