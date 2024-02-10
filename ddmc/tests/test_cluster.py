"""
Testing file for the clustering methods by data and sequence.
"""

import pytest
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture

from ddmc.clustering import DDMC
from ddmc.datasets import CPTAC, filter_incomplete_peptides


def test_wins():
    """Test that EMclustering is working by comparing with GMM clusters."""
    p_signal = filter_incomplete_peptides(
        CPTAC().get_p_signal(), sample_presence_ratio=1
    )
    model_ddmc = DDMC(n_components=2, seq_weight=0).fit(p_signal)
    model_gmm = GaussianMixture(n_components=2).fit(p_signal.values)

    similarity = cosine_similarity(model_gmm.means_, model_ddmc.transform().T)

    # only works for 2 clusters, check that the two clusters are matched up
    # either index-matching or otherwise
    diag = np.eye(2, dtype=bool)
    offdiag = ~diag

    assert np.all(similarity[diag] > 0.95) or np.all(similarity[offdiag] > 0.95)


@pytest.mark.parametrize("w", [0, 0.1, 10.0])
@pytest.mark.parametrize("ncl", [2, 5, 25])
@pytest.mark.parametrize("distance_method", ["PAM250", "Binomial"])
def test_clusters(w, ncl, distance_method):
    p_signal = filter_incomplete_peptides(
        CPTAC().get_p_signal(), sample_presence_ratio=1
    )
    model = DDMC(ncl, seq_weight=w, distance_method=distance_method).fit(p_signal)

    # Assert that we got a reasonable result
    assert np.all(np.isfinite(model.scores_))
    assert np.all(np.isfinite(model.seq_scores_))


@pytest.mark.parametrize("distance_method", ["PAM250", "Binomial"])
def test_ClusterVar(distance_method):
    """Test minimum variance of output cluster centers"""
    p_signal = filter_incomplete_peptides(
        CPTAC().get_p_signal(), sample_presence_ratio=1
    )

    model = DDMC(n_components=6, seq_weight=3, distance_method=distance_method).fit(
        p_signal
    )
    centers = model.transform()

    # Get pairwise cluster distances
    dists = cdist(centers.T, centers.T)
    np.fill_diagonal(dists, 10.0)

    # Assert that all the clusters are at least euclidean distance 1 away
    assert np.all(dists > 1.0)
