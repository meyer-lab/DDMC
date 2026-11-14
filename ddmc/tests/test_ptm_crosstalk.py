import numpy as np
import pandas as pd
import pytest

from ddmc.binomial import (
    Binomial,
    GenerateBinarySeqID,
    PTM_AAlist,
    fast_position_weight_matrix,
    normalize_ptm_sequences,
)
from ddmc.clustering import DDMC
from ddmc.motifs import annotate_flanking_ptms
from ddmc.pam250 import PAM250, get_pam250_scores


def test_normalize_ptm_sequences_and_encoding():
    seqs = ["AAAAGsGPsPP", "AARkRsAAAAA", "AAAAAtAAAAk"]
    norm_ptm = normalize_ptm_sequences(seqs, include_ptms=True)
    assert list(norm_ptm) == ["AAAAGSGPsPP", "AARkRSAAAAA", "AAAAATAAAAk"]

    norm_plain = normalize_ptm_sequences(seqs, include_ptms=False)
    assert list(norm_plain) == ["AAAAGSGPSPP", "AARKRSAAAAA", "AAAAATAAAAK"]

    onehot = GenerateBinarySeqID(seqs, include_ptms=True)
    assert onehot.shape == (3, len(PTM_AAlist), 11)
    assert onehot[0, PTM_AAlist.index("S"), 5]
    assert onehot[0, PTM_AAlist.index("s"), 8]
    assert onehot[1, PTM_AAlist.index("k"), 3]
    assert onehot[2, PTM_AAlist.index("T"), 5]
    assert onehot[2, PTM_AAlist.index("k"), 10]
    np.testing.assert_array_equal(onehot.sum(axis=1), np.ones((3, 11)))

    pwm = fast_position_weight_matrix(seqs, include_ptms=True)
    assert pwm.shape == (len(PTM_AAlist), 11)
    np.testing.assert_allclose(pwm.sum(axis=0), np.ones(11))


def test_pam250_ptm_invariants_and_priming_affinity():
    seqs = ["AAAAGsGPsPP", "AAAAGsGPSPP", "AARkRsAAAAA", "AARKRsAAAAA"]
    scores = get_pam250_scores(seqs, include_ptms=True)
    np.testing.assert_array_equal(scores, scores.T)
    for i in range(len(seqs)):
        assert scores[i, i] >= scores[i].max()
    # Primed self-score exceeds cross-score with unprimed isoform
    assert scores[0, 0] > scores[0, 1]
    assert scores[2, 2] > scores[2, 3]

    pam = PAM250(np.array(seqs), include_ptms=True)
    weights = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    pam.from_summaries(weights)
    assert pam.logWeights.shape == (4, 2)
    assert np.all(np.isfinite(pam.logWeights))


def test_binomial_distinguishes_primed_vs_unprimed_isoforms():
    seqs = np.array(["AAAAGsGPsPP"] * 6 + ["AAAAGsGPSPP"] * 6)
    model = Binomial(seqs, include_ptms=True)
    weights = np.zeros((12, 2))
    weights[:6, 0] = 1.0
    weights[6:, 1] = 1.0
    model.from_summaries(weights)

    # Primed peptides score higher in cluster 0; unprimed in cluster 1
    assert model.logWeights[0, 0] > model.logWeights[0, 1]
    assert model.logWeights[6, 1] > model.logWeights[6, 0]


def test_annotate_flanking_ptms_from_co_measured_sites():
    # Overlapping window:
    # peptide 0 centered at pos 5: "AAAAGsGPSPP" (pos 8 is S)
    # peptide 1 centered at pos 8 of peptide 0 -> shifted left by 3: "AGSGPsPPAAA"
    # acetylated K at pos 2 of peptide 2 ("AAKRRsAAAAA") -> shifted right by 3: "LLLAAkRRAAA"
    df = pd.DataFrame(
        {"s1": [1.0, 2.0, -1.0], "s2": [1.5, 2.5, -1.5]},
        index=["AAAAGsGPSPP", "AGSGPsPPAAA", "LLAAKsAAAAA"],
    )
    # For peptide 2 ("LLAAKsAAAAA"), K is at pos 4 (d = -1):
    # seq[:11 + d] = seq[:10] = "LLAAKSAAAA" == acetyl[-d:] = acetyl[1:]
    acetyl_motifs = ["XLLAAkSAAAA"]

    annotated = annotate_flanking_ptms(df, acetyl_motifs=acetyl_motifs)
    assert annotated.index[0] == "AAAAGsGPsPP"
    assert annotated.index[1] == "AGsGPsPPAAA"
    assert annotated.index[2] == "LLAAksAAAAA"


@pytest.mark.parametrize("distance_method", ["Binomial", "PAM250"])
def test_ddmc_separates_primed_and_unprimed_isoforms_on_held_out_samples(
    distance_method,
):
    rng = np.random.default_rng(7)
    n_per = 15
    # Exact same 11-AA backbone, differing only by flanking +3 pS vs unmodified S
    primed_seqs = ["AAAAGsGPsPP"] * n_per
    unprimed_seqs = ["AAAAGsGPSPP"] * n_per
    seqs = primed_seqs + unprimed_seqs

    # Training samples (4 cols) are masked/noisy for the last 2 test peptides;
    # test samples (2 cols) show divergent signaling (+3 vs -3).
    X = np.vstack(
        [
            rng.normal(loc=3.0, scale=0.4, size=(n_per, 6)),
            rng.normal(loc=-3.0, scale=0.4, size=(n_per, 6)),
        ]
    )
    df = pd.DataFrame(X, index=seqs, columns=[f"c{j}" for j in range(6)])

    model_ptm = DDMC(
        n_components=2,
        seq_weight=5.0,
        distance_method=distance_method,
        include_ptms=True,
        random_state=0,
    ).fit(df.iloc[:, :4])

    model_collapsed = DDMC(
        n_components=2,
        seq_weight=5.0,
        distance_method=distance_method,
        include_ptms=False,
        random_state=0,
    ).fit(df.iloc[:, :4])

    # Fold in using ONLY sequence (cols=[]) to isolate the sequence prior
    gamma_ptm = model_ptm.fold_in(df, cols=[])
    gamma_col = model_collapsed.fold_in(df, cols=[])

    # Collapsed 20-AA model assigns identical sequence probabilities to both isoforms
    np.testing.assert_allclose(gamma_col[0], gamma_col[n_per], atol=1e-6)
    # PTM-aware model cleanly separates primed vs unprimed isoforms
    assert np.argmax(gamma_ptm[0]) != np.argmax(gamma_ptm[n_per])

    # Verify get_pssms works cleanly even with lowercase flanking PTM tokens
    names, pssms = model_ptm.get_pssms()
    assert len(names) == 2
    assert pssms.shape == (2, 20, 11)
