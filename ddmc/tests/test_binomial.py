import numpy as np
import pytest

from ddmc.binomial import (
    AAlist,
    BackgProportions,
    Binomial,
    CountPsiteTypes,
    GenerateBinarySeqID,
    fast_position_weight_matrix,
    position_weight_matrix,
)


def test_fast_position_weight_matrix_matches_biopython():
    seqs = ["AAAAASAAAAA", "AAAAASAAAAA", "RRRRRTRRRRR"]
    fast = fast_position_weight_matrix(seqs)
    slow = np.array(list(position_weight_matrix(seqs).values()))

    assert fast.shape == (len(AAlist), 11)
    np.testing.assert_allclose(fast, slow, atol=1e-6)
    # Every position's frequencies should sum to 1
    np.testing.assert_allclose(fast.sum(axis=0), np.ones(11))


def test_fast_position_weight_matrix_enriches_observed_aa():
    seqs = ["AAAAASAAAAA"] * 10
    pwm = fast_position_weight_matrix(seqs)
    a_idx = AAlist.index("A")
    other_idx = AAlist.index("R")
    assert pwm[a_idx, 0] > pwm[other_idx, 0]


def test_generate_binary_seq_id():
    seqs = ["AAAAASAAAAA", "RAAAASAAAAA"]
    onehot = GenerateBinarySeqID(seqs)

    assert onehot.shape == (2, len(AAlist), 11)
    assert onehot[0, AAlist.index("A"), 0]
    assert not onehot[0, AAlist.index("R"), 0]
    assert onehot[1, AAlist.index("R"), 0]
    # Exactly one amino acid true per position
    assert np.all(onehot.sum(axis=1) == 1)


def test_generate_binary_seq_id_handles_lowercase_phosphoacceptor():
    onehot = GenerateBinarySeqID(["AAAAAsAAAAA"])
    assert onehot[0, AAlist.index("S"), 5]


@pytest.mark.parametrize(
    "seqs,expected",
    [
        (["AAAAAYAAAAA"], (1, 0, 0)),
        (["AAAAASAAAAA"], (0, 1, 0)),
        (["AAAAATAAAAA"], (0, 0, 1)),
        (["AAAAAyAAAAA", "AAAAAsAAAAA", "AAAAAtAAAAA"], (1, 1, 1)),
    ],
)
def test_count_psite_types(seqs, expected):
    assert CountPsiteTypes(np.array(seqs)) == expected


def test_backg_proportions_respects_max_counts_and_centers():
    # +/- 7 AA reference sequences (length 15), phosphoacceptor at index 7
    refseqs = [
        "AAAAAAAyAAAAAAA",
        "AAAAAAAyAAAAAAA",
        "AAAAAAAsAAAAAAA",
        "AAAAAAAtAAAAAAA",
        "AAAAAAAtAAAAAAA",
        "AAAAAAAtAAAAAAA",
    ]

    motifs = BackgProportions(refseqs, pYn=1, pSn=1, pTn=2)

    assert len(motifs) == 1 + 1 + 2
    assert all(len(m) == 11 for m in motifs)
    centers = [m[5] for m in motifs]
    assert centers == ["Y", "S", "T", "T"]


def test_backg_proportions_ignores_non_phosphoacceptor_center():
    refseqs = ["AAAAAAAAAAAAAAA"]  # center is "A", not y/t/s
    assert BackgProportions(refseqs, pYn=5, pSn=5, pTn=5) == []


def test_binomial_from_summaries_prefers_matching_cluster():
    seqs = np.array(["AAAAASAAAAA"] * 5 + ["RRRRRSRRRRR"] * 5)
    model = Binomial(seqs)

    # Two clusters: first gets all-A sequences, second gets all-R sequences
    weights = np.zeros((10, 2))
    weights[:5, 0] = 1.0
    weights[5:, 1] = 1.0
    model.from_summaries(weights)

    assert model.logWeights.shape == (10, 2)
    assert np.all(np.isfinite(model.logWeights))
    # Each sequence should score at least as well under its own cluster
    assert np.all(model.logWeights[:5, 0] >= model.logWeights[:5, 1])
    assert np.all(model.logWeights[5:, 1] >= model.logWeights[5:, 0])
