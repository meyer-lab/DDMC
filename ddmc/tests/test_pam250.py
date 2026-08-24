import numpy as np

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
