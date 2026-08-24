import numpy as np

from ddmc.binomial import AAlist
from ddmc.motifs import compute_control_pssm, get_keys_by_value, make_motif


def test_get_keys_by_value_finds_matches():
    d = {"PROT1": "AAABBBCCC", "PROT2": "DDDEEEFFF", "PROT3": "XXABBBXX"}
    assert sorted(get_keys_by_value(d, "BBB")) == ["PROT1", "PROT3"]


def test_get_keys_by_value_no_matches():
    d = {"PROT1": "AAABBBCCC"}
    assert get_keys_by_value(d, "ZZZ") == []


def test_make_motif_extracts_and_lowercases_center():
    up_seq = "MKVLAASTPPQRWY"
    # Phosphosite is the "S" at 0-indexed position 6
    motif, pidx = make_motif(
        up_seq, "AASTP", 5, ps_protein_idx=6, center_motif_idx=2, DoS_idx=None
    )

    assert motif == "KVLAAsTPPQR"
    assert motif[5] == "s"
    assert pidx == ["S7-p"]


def test_make_motif_pads_near_sequence_start():
    up_seq = "MKSTPQRWYVL"
    motif, pidx = make_motif(
        up_seq, "MKS", 5, ps_protein_idx=2, center_motif_idx=2, DoS_idx=None
    )

    assert motif == "---MKsTPQRW"
    assert pidx == ["S3-p"]


def test_compute_control_pssm_shape_and_values():
    seqs = ["AAAAASAAAAA"] * 5 + ["RRRRRSRRRRR"] * 5
    with np.errstate(divide="ignore"):
        pssm = compute_control_pssm(seqs)

    assert pssm.shape == (len(AAlist), 11)
    assert np.all(np.isfinite(pssm))
    # A and R are equally represented at every position, so their log2
    # enrichment should be identical (and other AAs should show as -inf
    # -> nan_to_num'd depletion, i.e. very negative or zero).
    a_idx, r_idx = AAlist.index("A"), AAlist.index("R")
    np.testing.assert_allclose(pssm[a_idx], pssm[r_idx])
