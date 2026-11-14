import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from ddmc.clustering import DDMC, get_pspl_pssm_distances
from ddmc.motifs import KinToPhosphotypeDict, get_kinome_metadata, get_pspls


def test_get_pspls_loads_full_human_kinome_atlas():
    kinases, pspls = get_pspls()
    assert kinases.shape == (404,)
    assert pspls.shape == (404, 20, 9)
    assert np.all(np.isfinite(pspls))

    for kin in kinases:
        assert kin in KinToPhosphotypeDict
        assert KinToPhosphotypeDict[kin] in ("S/T", "Y")

    meta = get_kinome_metadata()
    assert len(meta) == 404
    assert {"KINASE", "TYPE", "SUBTYPE", "FAMILY", "DUAL_SPECIFICITY"}.issubset(
        meta.columns
    )


def test_get_pspls_legacy_and_invalid_source():
    kinases_legacy, pspls_legacy = get_pspls(source="legacy")
    assert kinases_legacy.shape[0] == 55
    assert pspls_legacy.shape == (55, 20, 9)
    for kin in kinases_legacy:
        assert kin in KinToPhosphotypeDict

    with pytest.raises(ValueError, match="Unknown PSPL source"):
        get_pspls(source="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("metric", ["frobenius", "js", "kl", "cosine", "zscore"])
def test_get_pspl_pssm_distances_metrics(metric):
    pspls = np.array(
        [
            np.arange(180, dtype=np.float64).reshape(20, 9) / 60.0,
            np.ones((20, 9), dtype=np.float64),
        ]
    )
    # Build PSSMs whose 9 flanking positions (excluding 5 and 10) match pspls
    pssms = np.zeros((2, 20, 11), dtype=np.float64)
    flanking_cols = [0, 1, 2, 3, 4, 6, 7, 8, 9]
    pssms[0, :, flanking_cols] = pspls[0].T
    pssms[1, :, flanking_cols] = pspls[1].T

    dists = get_pspl_pssm_distances(pspls, pssms, metric=metric)
    assert dists.shape == (2, 2)
    assert np.all(np.isfinite(dists))
    if metric != "zscore":
        assert dists[0, 0] == pytest.approx(0.0, abs=1e-9)
        assert dists[1, 1] == pytest.approx(0.0, abs=1e-9)
    else:
        assert dists[0, 0] < dists[1, 0]


def test_get_pspl_pssm_distances_rejects_bad_metric():
    pspls = np.zeros((1, 20, 9))
    pssms = np.zeros((1, 20, 11))
    with pytest.raises(ValueError, match="Unsupported metric"):
        get_pspl_pssm_distances(pspls, pssms, metric="bogus")  # type: ignore[arg-type]


def test_predict_upstream_kinases_with_full_atlas(p_signal):
    model = DDMC(n_components=3, seq_weight=5.0, random_state=0).fit(p_signal.iloc[:60])
    dists = model.predict_upstream_kinases(metric="js")
    assert dists.shape == (404, model.get_nonempty_clusters().size)
    assert np.all(np.isfinite(dists.values))
    assert "ERK2" in dists.index
    assert "AKT1" in dists.index
    assert "SRC" in dists.index


def test_deconvolves_degenerate_proline_directed_substrates():
    """Two substrate groups sharing the `[S/T]-P` core motif but regulated by
    distinct abundance dynamics should be cleanly separated by DDMC.
    """
    rng = np.random.default_rng(7)
    n_each = 25
    # Both groups share central S and +1 P (`[S/T]-P` proline-directed motif)
    seqs = ["AAAAPsPAAAA"] * n_each + ["AAAAPsPAAAA"] * n_each
    c0 = np.tile(np.array([2.0, 2.0, -2.0, -2.0, 2.0, 2.0]), (n_each, 1))
    c1 = np.tile(np.array([-2.0, -2.0, 2.0, 2.0, -2.0, -2.0]), (n_each, 1))
    data = np.vstack([c0, c1]) + rng.normal(scale=0.3, size=(2 * n_each, 6))
    df = pd.DataFrame(data, index=seqs, columns=[f"c{i}" for i in range(6)])

    model = DDMC(n_components=2, seq_weight=2.0, random_state=0).fit(df)
    true_labels = np.array([0] * n_each + [1] * n_each)
    assert adjusted_rand_score(true_labels, model.labels()) == pytest.approx(1.0)
