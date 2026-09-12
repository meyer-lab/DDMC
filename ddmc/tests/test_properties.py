"""Property-based tests (using Hypothesis) for DDMC's core numeric helpers.

These tests generate random-but-valid inputs and check invariants that
should hold for *any* such input, rather than a single hand-picked example.
They are meant to complement, not replace, the example-based tests in the
other `test_*.py` modules.
"""

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from ddmc.binomial import (
    AAlist,
    CountPsiteTypes,
    GenerateBinarySeqID,
    fast_position_weight_matrix,
    position_weight_matrix,
)
from ddmc.logistic_regression import normalize_cluster_centers
from ddmc.motifs import compute_control_pssm
from ddmc.pam250 import get_pam250_scores

SEQ_LEN = 11
# Amino acids that carry a nonzero pseudocount and so are always well
# defined position-weight-matrix inputs.
aa_chars = st.sampled_from(AAlist)
peptide = st.text(alphabet=AAlist, min_size=SEQ_LEN, max_size=SEQ_LEN)
peptide_list = st.lists(peptide, min_size=1, max_size=8)

# Uniformly random peptides almost never repeat one high-scoring residue
# (e.g. tryptophan) at every position, so a purely random `peptide`
# strategy is very unlikely to ever exercise the int8-overflow-prone
# region of `get_pam250_scores` (see tests below). Bias part of the input
# space towards single-residue-repeat sequences, which is where that
# overflow actually happens.
skewed_peptide = st.one_of(peptide, aa_chars.map(lambda aa: aa * SEQ_LEN))
skewed_peptide_list = st.lists(skewed_peptide, min_size=1, max_size=8)


# ---------------------------------------------------------------------------
# get_pam250_scores (ddmc.pam250)
# ---------------------------------------------------------------------------


@given(skewed_peptide_list)
@settings(max_examples=300)
def test_pam250_scores_are_symmetric(seqs):
    scores = get_pam250_scores(seqs)
    np.testing.assert_array_equal(scores, scores.T)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: get_pam250_scores overflows its int8 accumulator for "
    "highly self-similar peptides (e.g. poly-W/poly-C); see docstring.",
)
@given(skewed_peptide)
@example("W" * SEQ_LEN)
@example("C" * SEQ_LEN)
@settings(max_examples=300)
def test_pam250_self_score_matches_diagonal_sum(seq):
    """A sequence's PAM250 score against itself is, by construction, the sum
    of the substitution matrix's self-similarity values for each of its
    residues. This must hold regardless of amino acid composition.

    BUG: `get_pam250_scores` accumulates scores in an `int8` array. Highly
    self-similar residues (e.g. tryptophan, self-score 17) push the summed
    per-sequence score above 127 for realistic peptide lengths, silently
    wrapping around to a negative number instead of raising or saturating.
    """
    from Bio.Align import substitution_matrices

    pam250 = substitution_matrices.load("PAM250")
    expected = sum(pam250[aa, aa] for aa in seq)

    scores = get_pam250_scores([seq])
    assert scores[0, 0] == expected


@pytest.mark.xfail(
    strict=True,
    reason="BUG: get_pam250_scores int8 overflow can push a sequence's "
    "self-score below its off-diagonal scores; see docstring.",
)
@given(skewed_peptide_list)
@example(["W" * SEQ_LEN, "A" * SEQ_LEN])
@settings(max_examples=300)
def test_pam250_self_score_is_the_max_for_the_row(seqs):
    """A sequence should never be *less* similar to itself than to any other
    (potentially different) sequence in the set -- PAM250 self-substitution
    scores are the largest entry in every row/column of the matrix.

    This is also broken by the int8 overflow above: an overflowed diagonal
    entry can end up *below* the sequence's off-diagonal scores.
    """
    scores = get_pam250_scores(seqs)
    for i in range(len(seqs)):
        assert scores[i, i] >= scores[i].max()


# ---------------------------------------------------------------------------
# fast_position_weight_matrix vs. the Biopython-backed reference
# implementation (ddmc.binomial)
# ---------------------------------------------------------------------------


@given(peptide_list)
@settings(max_examples=200)
def test_fast_pwm_matches_biopython_reference(seqs):
    fast = fast_position_weight_matrix(seqs)
    slow = np.array(list(position_weight_matrix(seqs).values()))

    assert fast.shape == (len(AAlist), SEQ_LEN)
    np.testing.assert_allclose(fast, slow, atol=1e-6)


@given(peptide_list)
@settings(max_examples=200)
def test_fast_pwm_columns_sum_to_one(seqs):
    pwm = fast_position_weight_matrix(seqs)
    np.testing.assert_allclose(pwm.sum(axis=0), np.ones(SEQ_LEN))
    assert np.all(pwm >= 0)


# ---------------------------------------------------------------------------
# GenerateBinarySeqID (ddmc.binomial)
# ---------------------------------------------------------------------------


@given(peptide_list)
@settings(max_examples=200)
def test_generate_binary_seq_id_is_one_hot(seqs):
    onehot = GenerateBinarySeqID(seqs)
    assert onehot.shape == (len(seqs), len(AAlist), SEQ_LEN)
    # Exactly one amino acid true per position, for every sequence.
    np.testing.assert_array_equal(onehot.sum(axis=1), np.ones((len(seqs), SEQ_LEN)))


@given(peptide)
@settings(max_examples=200)
def test_generate_binary_seq_id_roundtrips_sequence(seq):
    onehot = GenerateBinarySeqID([seq])[0]
    recovered = "".join(AAlist[np.argmax(onehot[:, pos])] for pos in range(SEQ_LEN))
    assert recovered == seq


# ---------------------------------------------------------------------------
# CountPsiteTypes (ddmc.binomial)
# ---------------------------------------------------------------------------


@given(peptide_list)
@settings(max_examples=200)
def test_count_psite_types_accounts_for_every_sequence(seqs):
    pY, pS, pT = CountPsiteTypes(np.array(seqs))
    center = SEQ_LEN // 2
    centers = [s[center].upper() for s in seqs]
    assert pY == centers.count("Y")
    assert pS == centers.count("S")
    assert pT == centers.count("T")
    # Every non-S/T/Y-centered sequence is simply not counted anywhere --
    # the three counts should never exceed the number of input sequences.
    assert pY + pS + pT <= len(seqs)


# ---------------------------------------------------------------------------
# compute_control_pssm (ddmc.motifs)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG: compute_control_pssm crashes on the lowercase phosphoacceptor "
    "that BackgroundSeqs (its real caller) always produces; see docstring.",
)
@given(
    st.lists(
        st.text(alphabet=AAlist, min_size=SEQ_LEN, max_size=SEQ_LEN),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=200)
def test_compute_control_pssm_matches_uppercase_input(seqs):
    """`ddmc.binomial.BackgroundSeqs` documents that it lowercases the
    central phosphoacceptor of every background sequence it returns, and
    `DDMC.get_pssms(PsP_background=True)` feeds that output straight into
    `compute_control_pssm` (ddmc/clustering.py). So a lowercase center
    character at position 5 is a realistic, in-band input for this
    function -- not an out-of-domain one.

    BUG: `compute_control_pssm` looks up each residue with
    `AAlist.index(aa)` without upper-casing it first, so any sequence with
    a lowercase phosphoacceptor (exactly what `BackgroundSeqs` produces)
    raises `ValueError: '<aa>' is not in list` instead of being scored.
    """
    lowercased = [seq[:5] + seq[5].lower() + seq[6:] for seq in seqs]

    upper_result = compute_control_pssm(seqs)
    lower_result = compute_control_pssm(lowercased)

    np.testing.assert_allclose(lower_result, upper_result)


# ---------------------------------------------------------------------------
# normalize_cluster_centers (ddmc.logistic_regression)
# ---------------------------------------------------------------------------


finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: normalize_cluster_centers centers along the wrong axis "
    "(per-sample instead of per-cluster); see docstring.",
)
@given(
    st.integers(min_value=2, max_value=6).flatmap(
        lambda n_samples: st.lists(
            st.lists(finite_floats, min_size=n_samples, max_size=n_samples),
            min_size=1,
            max_size=5,
        ).map(lambda cols: np.array(cols).T)
    )
)
@example(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).T)
@settings(max_examples=200)
def test_normalize_cluster_centers_zero_means_each_cluster(centers):
    """`normalize_cluster_centers`'s docstring says it centers each cluster
    (column) to zero mean *across samples* (rows) -- the usual
    pre-classification feature standardization.

    BUG: it actually does the opposite. `StandardScaler(...).fit_transform`
    always centers each *column* of the array it is given across that
    array's rows; wrapping the call in `centers.T ... .T` makes it center
    each *sample* (row of `centers`) across that sample's clusters instead,
    leaving the per-cluster (per-column) mean across samples unchanged (and
    nonzero, whenever the input wasn't already centered that way). The
    `.T`/`.T` should not have been added -- `StandardScaler(with_std=False
    ).fit_transform(centers)` directly would do what the docstring
    describes.
    """
    normalized = normalize_cluster_centers(centers)
    assert normalized.shape == centers.shape
    np.testing.assert_allclose(
        normalized.mean(axis=0), np.zeros(centers.shape[1]), atol=1e-6
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
