import numpy as np
import pytest

from ddmc.pam250 import PAM250, get_pam250_scores


def test_get_pam250_scores_symmetric_and_self_max():
    seqs = ["AAAAAAAAAAA", "RRRRRRRRRRR", "AAAAAAAAAAR"]
    scores = get_pam250_scores(seqs)

    assert scores.shape == (3, 3)
    np.testing.assert_array_equal(scores, scores.T)
    # A sequence should score at least as well against itself as against
    # any other sequence in the set.
    for i in range(len(seqs)):
        assert scores[i, i] >= scores[i].max() - 1e-9


def test_pam250_from_summaries_prefers_matching_cluster():
    seqs = ["AAAAAAAAAAA"] * 3 + ["RRRRRRRRRRR"] * 3
    model = PAM250(seqs)

    weights = np.zeros((6, 2))
    weights[:3, 0] = 1.0
    weights[3:, 1] = 1.0
    model.from_summaries(weights)

    assert model.logWeights.shape == (6, 2)
    assert np.all(model.logWeights[:3, 0] >= model.logWeights[:3, 1])
    assert np.all(model.logWeights[3:, 1] >= model.logWeights[3:, 0])


def test_pam250_from_summaries_avoids_empty_cluster_division_by_zero():
    seqs = ["AAAAAAAAAAA", "RRRRRRRRRRR"]
    model = PAM250(seqs)

    # Second cluster gets no weight at all
    weights = np.array([[1.0, 0.0], [1.0, 0.0]])
    model.from_summaries(weights)

    assert np.all(np.isfinite(model.logWeights))


def test_pam250_ema_matches_full_batch():
    seqs = ["AAAAAAAAAAA"] * 3 + ["RRRRRRRRRRR"] * 3
    weights = np.zeros((6, 2))
    weights[:3, 0] = 1.0
    weights[3:, 1] = 1.0

    ref = PAM250(seqs)
    ref.from_summaries(weights)

    ema = PAM250(seqs)
    ema.init_ema(2)
    ema.update_ema(np.arange(6), weights, alpha=1.0, scale=1.0)
    ema.finalize_ema()
    ema.compute_log_weights()

    np.testing.assert_allclose(ema.logWeights, ref.logWeights, atol=1e-6)
    np.testing.assert_allclose(
        ema.estimate_log_prob_batch(np.array([0, 4])),
        ref.logWeights[[0, 4]],
        atol=1e-6,
    )


def test_pam250_split_cluster():
    seqs = ["AAAAAAAAAAA"] * 3 + ["RRRRRRRRRRR"] * 3
    ema = PAM250(seqs)
    ema.init_ema(2)
    weights = np.zeros((6, 2))
    weights[:, 0] = 1.0
    ema.update_ema(np.arange(6), weights, alpha=1.0, scale=1.0)

    ema.split_cluster(donor=0, target=1)
    assert ema._n_ema[0] == pytest.approx(ema._n_ema[1])
    np.testing.assert_allclose(ema._m_ema[:, 0], ema._m_ema[:, 1])
