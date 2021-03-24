"""
Testing file for minimum variance across clusters using CPTAC data.
"""

import pytest
import numpy as np
import pandas as pd
from collections import Counter
from ..clustering import MassSpecClustering
from ..pre_processing import filter_NaNpeptides

X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
X = filter_NaNpeptides(X, tmt=25)
d = X.select_dtypes(include=['float64']).T
i = X.select_dtypes(include=['object'])
preMotifSet = ["ABL", "EGFR", "ALK", "SRC", "YES"]


@pytest.mark.parametrize("distance_method", ["PAM250", "Binomial", "PAM250_fixed"])
def test_ClusterVar(distance_method):
    """Test minimum variance of output cluster centers """
    model = MassSpecClustering(i, 5, SeqWeight=3, distance_method=distance_method, pre_motifs=preMotifSet).fit(X=d)
    centers = model.transform()
    nRows = 5
    for row in range(nRows):
        vals = np.round(centers[row, :], 3)
        repeats = np.array(list(Counter(vals).values()))
        assert np.all(repeats < 3), "More than 3 repeated values."