"""
Testing file for the clustering methods by data and sequence.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture

from ddmc.clustering import DDMC, get_pspl_pssm_distances


def test_wins(p_signal):
    """Test that EMclustering is working by comparing with GMM clusters."""
    model_ddmc = DDMC(n_components=2, seq_weight=0).fit(p_signal)
    model_gmm = GaussianMixture(n_components=2).fit(p_signal.values)

    similarity = cosine_similarity(model_gmm.means_, model_ddmc.transform().T)

    # only works for 2 clusters, check that the two clusters are matched up
    # either index-matching or otherwise
    diag = np.eye(2, dtype=bool)
    offdiag = ~diag

    assert np.all(similarity[diag] > 0.95) or np.all(similarity[offdiag] > 0.95)


@pytest.mark.parametrize("w", [0, 10.0])
@pytest.mark.parametrize("ncl", [2, 25])
@pytest.mark.parametrize("distance_method", ["PAM250", "Binomial"])
def test_clusters(p_signal, w, ncl, distance_method):
    model = DDMC(ncl, seq_weight=w, distance_method=distance_method).fit(p_signal)

    # Assert that we got a reasonable result
    assert np.all(np.isfinite(model.scores_))
    assert np.all(np.isfinite(model.seq_scores_))


@pytest.mark.parametrize("distance_method", ["PAM250", "Binomial"])
def test_ClusterVar(p_signal, distance_method):
    """Test minimum variance of output cluster centers"""
    model = DDMC(n_components=6, seq_weight=3, distance_method=distance_method).fit(
        p_signal
    )
    centers = model.transform()

    # Get pairwise cluster distances
    dists = cdist(centers.T, centers.T)
    np.fill_diagonal(dists, 10.0)

    # Assert that all the clusters are at least euclidean distance 1 away
    assert np.all(dists > 1.0)


@pytest.fixture(scope="session")
def fitted_model(p_signal):
    """A small, cheap-to-fit model shared across tests that only inspect
    post-fit accessors (rather than clustering quality itself)."""
    return DDMC(n_components=3, seq_weight=1.0).fit(p_signal.iloc[:50])


def test_predict_and_labels_agree(fitted_model):
    assert np.array_equal(fitted_model.predict(), fitted_model.labels())
    assert fitted_model.labels().shape == (fitted_model.p_signal.shape[0],)
    assert np.all(fitted_model.labels() < fitted_model.n_components)


def test_get_nonempty_clusters_and_has_empty_clusters(fitted_model):
    nonempty = fitted_model.get_nonempty_clusters()
    assert set(nonempty) == set(np.unique(fitted_model.labels()))
    assert fitted_model.has_empty_clusters() == (
        nonempty.size != fitted_model.n_components
    )


def test_score_is_finite_float(fitted_model):
    score = fitted_model.score()
    assert isinstance(score, float)
    assert np.isfinite(score)


def test_transform_as_df(fitted_model):
    centers = fitted_model.transform(as_df=True)
    assert isinstance(centers, pd.DataFrame)
    assert list(centers.index) == list(fitted_model.p_signal.columns)
    assert list(centers.columns) == list(range(fitted_model.n_components))
    np.testing.assert_allclose(centers.values, fitted_model.transform())


def test_impute_fills_missing_and_matches_shape(fitted_model):
    imputed = fitted_model.impute()
    assert imputed.shape == fitted_model.p_signal.shape
    assert np.all(np.isfinite(imputed))


def test_impute_uses_assigned_cluster_center():
    """Regression test for #562: `DDMC.impute()` previously indexed
    `centers[:, labels[ii] - 1]`, shifting every peptide's assigned cluster by
    -1 and filling missing entries from the wrong cluster's center."""
    rng = np.random.default_rng(0)
    seqs = ["AAAAAsAAAAA"] * 6 + ["RRRRRsRRRRR"] * 6
    data = np.vstack(
        [
            rng.normal(loc=10.0, scale=0.1, size=(6, 4)),
            rng.normal(loc=-10.0, scale=0.1, size=(6, 4)),
        ]
    )
    data[0, 1] = np.nan
    data[6, 2] = np.nan
    df = pd.DataFrame(data, index=seqs, columns=["s1", "s2", "s3", "s4"])

    model = DDMC(n_components=2, seq_weight=1.0, random_state=0).fit(df)
    imputed = model.impute()
    labels = model.labels()
    centers = model.transform()

    assert imputed.iloc[0, 1] == pytest.approx(centers[1, labels[0]])
    assert imputed.iloc[6, 2] == pytest.approx(centers[2, labels[6]])


def test_get_pssms_uses_matching_cluster_scores():
    """Regression test for #562: `DDMC.get_pssms()` previously accumulated
    `self.scores_[jj, ii - 1]` for cluster `ii` and only populated `back_pssm`
    when `ii == 1` (leaving cluster 0's flanking PSSM columns all-zero when
    `PsP_background=False`)."""
    rng = np.random.default_rng(1)
    seqs = ["AAAAAsAAAAA"] * 8 + ["RRRRRsRRRRR"] * 8
    data = np.vstack(
        [
            rng.normal(loc=5.0, scale=0.05, size=(8, 3)),
            rng.normal(loc=-5.0, scale=0.05, size=(8, 3)),
        ]
    )
    df = pd.DataFrame(data, index=seqs, columns=["c1", "c2", "c3"])

    model = DDMC(n_components=2, seq_weight=10.0, random_state=0).fit(df)
    clusters, pssms = model.get_pssms(PsP_background=False)

    assert list(clusters) == [0, 1]
    a_cluster = int(model.labels()[0])
    r_cluster = int(model.labels()[8])
    assert a_cluster != r_cluster

    from ddmc.binomial import AAlist

    a_idx, r_idx = AAlist.index("A"), AAlist.index("R")
    # Flanking positions (all positions except 5) in `a_cluster` must be
    # enriched for 'A' (> 0) and not 'R', including when `a_cluster == 0`.
    flanking = [p for p in range(11) if p != 5]
    assert np.all(pssms[a_cluster, a_idx, flanking] > 0.5)
    assert np.all(pssms[a_cluster, r_idx, flanking] == 0.0)
    assert np.all(pssms[r_cluster, r_idx, flanking] > 0.5)
    assert np.all(pssms[r_cluster, a_idx, flanking] == 0.0)


def test_fit_rejects_non_string_sequence():
    p_signal = pd.DataFrame({"a": [1.0, 2.0]}, index=[1234567890123, "AAAAAAAAAAA"])
    with pytest.raises(AssertionError, match="is not a string"):
        DDMC(n_components=2, seq_weight=1.0).fit(p_signal)


def test_fit_rejects_wrong_length_sequence():
    p_signal = pd.DataFrame({"a": [1.0, 2.0]}, index=["SHORT", "AAAAAAAAAAA"])
    with pytest.raises(AssertionError, match="is of length"):
        DDMC(n_components=2, seq_weight=1.0).fit(p_signal)


def test_fit_rejects_invalid_amino_acid():
    p_signal = pd.DataFrame({"a": [1.0, 2.0]}, index=["AAAAAAAAAAA", "AAAAABAAAAA"])
    with pytest.raises(AssertionError, match="invalid characters"):
        DDMC(n_components=2, seq_weight=1.0).fit(p_signal)


def test_fit_rejects_non_numeric_column():
    p_signal = pd.DataFrame(
        {"a": ["not", "numeric"]}, index=["AAAAAAAAAAA", "SSSSSSSSSSS"]
    )
    with pytest.raises(AssertionError, match="should be numerical"):
        DDMC(n_components=2, seq_weight=1.0).fit(p_signal)


def test_gen_peptide_distances_rejects_bad_distance_method():
    model = DDMC(n_components=2, seq_weight=1.0, distance_method="bogus")
    with pytest.raises(ValueError, match="Wrong distance type"):
        model._gen_peptide_distances(["AAAAAAAAAAA", "SSSSSSSSSSS"], "bogus")


def test_get_pspl_pssm_distances_shape_and_self_distance():
    pssms = np.random.rand(2, 20, 11)
    # Build a PSPL that exactly matches the first pssm (after dropping the
    # phosphoacceptor columns 5 and 10) so its self-distance is zero.
    pspl_from_pssm = np.delete(pssms[0], [5, 10], axis=1)
    pspls = np.stack([pspl_from_pssm, np.zeros((20, 9))])

    dists = get_pspl_pssm_distances(pspls, pssms)

    assert dists.shape == (2, 2)
    assert dists[0, 0] == pytest.approx(0.0)


def test_get_pspl_pssm_distances_as_df():
    pssms = np.zeros((1, 20, 11))
    pspls = np.zeros((2, 20, 9))
    dists = get_pspl_pssm_distances(
        pspls, pssms, as_df=True, pssm_names=["cluster_0"], kinases=["K1", "K2"]
    )
    assert isinstance(dists, pd.DataFrame)
    assert list(dists.index) == ["K1", "K2"]
    assert list(dists.columns) == ["cluster_0"]


def test_emeans_eliminates_empty_clusters_under_high_seq_weight(p_signal):
    model = DDMC(
        n_components=25, seq_weight=100.0, distance_method="Binomial", random_state=5
    ).fit(p_signal)
    assert not model.has_empty_clusters()
    assert model.get_nonempty_clusters().size == 25


def test_emeans_schedule_scaling_laws():
    batch_size, alpha_0 = DDMC._emeans_schedule(n=30561, k=30, n_epochs=10)
    assert 30 <= batch_size <= 30561
    assert 0.0 < alpha_0 <= 1.0

    small_b, small_alpha = DDMC._emeans_schedule(n=12, k=2, n_epochs=10)
    assert small_b == 12
    assert 0.0 < small_alpha <= 1.0


def test_emeans_revive_dead_clusters():
    model = DDMC(n_components=3, seq_weight=0.0, random_state=0)
    model.reg_covar = 1e-6
    rng = np.random.RandomState(0)

    n_ema = np.array([100.0, 50.0, 0.1])
    means = np.array([[1.0, 2.0], [-1.0, -2.0], [0.0, 0.0]])
    vars_ = np.ones((3, 2))
    s_ema = n_ema[:, None] * means
    q_ema = n_ema[:, None] * (vars_ + means * means)

    model._revive_dead_clusters(
        n_ema, s_ema, q_ema, min_cluster_size=1.0, rng=rng, revive_seq=False
    )

    assert np.all(n_ema >= 1.0)
    assert float(n_ema.sum()) == pytest.approx(150.0)


def test_emeans_reproducible_with_random_state(p_signal):
    sub = p_signal.iloc[:100]
    m1 = DDMC(n_components=4, seq_weight=5.0, random_state=42).fit(sub)
    m2 = DDMC(n_components=4, seq_weight=5.0, random_state=42).fit(sub)
    np.testing.assert_allclose(m1.means_, m2.means_)
    np.testing.assert_allclose(m1.scores_, m2.scores_)
    assert m1.score() == pytest.approx(m2.score())
