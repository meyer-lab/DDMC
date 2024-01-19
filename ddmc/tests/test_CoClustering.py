"""
Testing file for the clustering methods by data and sequence.
"""

import pytest
import numpy as np
import pandas as pd
from ..clustering import DDMC
from ..pre_processing import filter_NaNpeptides


X = pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
X = filter_NaNpeptides(X, tmt=25)
data = X.select_dtypes(include=["float64"]).T
info = X.select_dtypes(include=["object"])


@pytest.mark.parametrize("distance_method", ["PAM250", "Binomial"])
def test_wins(distance_method):
    """Test that EMclustering is working by comparing with GMM clusters."""
    MSC = DDMC(info, 2, SeqWeight=0, distance_method=distance_method).fit(
        X=data
    )
    distances = MSC.wins(data)

    # assert that the distance to the same sequence weight is less
    assert distances[0] < 1.0
    assert distances[0] < distances[1]


@pytest.mark.parametrize("w", [0, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("ncl", [2, 5, 10, 25])
@pytest.mark.parametrize("distance_method", ["PAM250", "Binomial"])
def test_clusters(w, ncl, distance_method):
    """Test that EMclustering is working by comparing with GMM clusters."""
    MSC = DDMC(
        info, ncl, SeqWeight=w, distance_method=distance_method
    ).fit(X=data)

    # Assert that we got a reasonable result
    assert np.all(np.isfinite(MSC.scores_))
    assert np.all(np.isfinite(MSC.seq_scores_))
