"""
Testing file for minimum variance across clusters using CPTAC data.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from ..clustering import DDMC
from ..pre_processing import filter_NaNpeptides

X = pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
X = filter_NaNpeptides(X, tmt=25)
d = X.select_dtypes(include=["float64"]).T


@pytest.mark.parametrize("distance_method", ["PAM250", "Binomial"])
def test_ClusterVar(distance_method):
    """Test minimum variance of output cluster centers"""
    model = DDMC(X["Sequence"], 6, seq_weight=3, distance_method=distance_method).fit(
        X=d
    )
    centers = model.transform()

    # Get pairwise cluster distances
    dists = cdist(centers.T, centers.T)
    np.fill_diagonal(dists, 10.0)

    # Assert that all the clusters are at least euclidean distance 1 away
    assert np.all(dists > 1.0)
