import numpy as np
import pandas as pd
import pytest

from ddmc.bcv import bi_cross_validate, select_best_hyperparams
from ddmc.binomial import Binomial
from ddmc.clustering import DDMC
from ddmc.pam250 import PAM250


def _make_synthetic_dataset(
    n_per_cluster: int = 30, n_cols: int = 12, seed: int = 0
) -> pd.DataFrame:
    """Two well-separated clusters with distinct sequences and MS dynamics."""
    rng = np.random.default_rng(seed)
    seqs = ["AAAAASAAAAA"] * n_per_cluster + ["RRRRRSRRRRR"] * n_per_cluster
    # Give each cluster distinct phosphorylation profiles plus moderate noise
    c0 = np.tile(np.linspace(-2.0, 2.0, n_cols), (n_per_cluster, 1))
    c1 = np.tile(np.linspace(2.0, -2.0, n_cols), (n_per_cluster, 1))
    data = np.vstack([c0, c1]) + rng.normal(scale=0.5, size=(2 * n_per_cluster, n_cols))
    cols = [f"s{i}" for i in range(n_cols)]
    return pd.DataFrame(data, index=seqs, columns=cols)


def test_binomial_score_sequences_matches_training_and_unseen():
    seqs = np.array(["AAAAASAAAAA"] * 5 + ["RRRRRSRRRRR"] * 5)
    weights = np.zeros((10, 2))
    weights[:5, 0] = 1.0
    weights[5:, 1] = 1.0

    model = Binomial(seqs)
    model.from_summaries(weights)

    np.testing.assert_allclose(model.score_sequences(seqs), model.logWeights, atol=1e-6)
    unseen = model.score_sequences(["AAAAASAAAAA", "RRRRRSRRRRR"])
    assert unseen.shape == (2, 2)
    assert unseen[0, 0] > unseen[0, 1]
    assert unseen[1, 1] > unseen[1, 0]


def test_pam250_score_sequences_matches_training_and_unseen():
    seqs = ["AAAAASAAAAA"] * 4 + ["RRRRRSRRRRR"] * 4
    weights = np.zeros((8, 2))
    weights[:4, 0] = 1.0
    weights[4:, 1] = 1.0

    model = PAM250(seqs)
    model.from_summaries(weights)

    np.testing.assert_allclose(model.score_sequences(seqs), model.logWeights, atol=1e-6)
    unseen = model.score_sequences(["AAAAASAAAAA", "RRRRRSRRRRR"])
    assert unseen.shape == (2, 2)
    assert unseen[0, 0] > unseen[0, 1]
    assert unseen[1, 1] > unseen[1, 0]


def test_fold_in_partial_empty_and_missing():
    df = _make_synthetic_dataset(n_per_cluster=20, n_cols=8, seed=1)
    train_df = df.iloc[np.r_[0:15, 20:35]]
    test_df = df.iloc[np.r_[15:20, 35:40]].copy()
    test_df.iloc[0, 0] = np.nan

    model = DDMC(n_components=2, seq_weight=5.0, random_state=0).fit(train_df)

    # Full, partial (with NaN), and sequence-only (empty cols) fold-in
    for cols in (None, [0, 1, 2, 3], []):
        gamma = model.fold_in(test_df, cols=cols)
        assert gamma.shape == (10, 2)
        np.testing.assert_allclose(gamma.sum(axis=1), np.ones(10), atol=1e-6)
        assert np.all(np.isfinite(gamma))
        # First 5 test peptides belong to cluster A, last 5 to cluster R
        assert np.all(np.argmax(gamma[:5], axis=1) == np.argmax(gamma[0]))
        assert np.all(np.argmax(gamma[5:], axis=1) == np.argmax(gamma[5]))
        assert np.argmax(gamma[0]) != np.argmax(gamma[5])


@pytest.mark.parametrize("distance_method", ["Binomial", "PAM250"])
def test_bi_cross_validate_grid_and_selection(distance_method):
    df = _make_synthetic_dataset(n_per_cluster=20, n_cols=9, seed=2)
    res = bi_cross_validate(
        df,
        n_components=[1, 2],
        seq_weight=[0.0, 5.0],
        distance_method=distance_method,
        row_folds=3,
        col_folds=3,
        random_state=0,
        n_epochs=6,
    )

    assert list(res.columns) == [
        "n_components",
        "seq_weight",
        "log_likelihood",
        "mse",
        "r2",
        "n_obs",
    ]
    assert len(res) == 4
    assert np.all(np.isfinite(res[["log_likelihood", "mse", "r2"]]))
    # n_obs should equal total entries evaluated across the 3x3 grid
    assert res["n_obs"].nunique() == 1
    assert res["n_obs"].iloc[0] == df.size

    best_k_ll, _ = select_best_hyperparams(res, metric="log_likelihood")
    best_k_mse, _ = select_best_hyperparams(res, metric="mse")
    best_k_r2, _ = select_best_hyperparams(res, metric="r2")
    assert best_k_ll == 2
    assert best_k_mse == 2
    assert best_k_r2 == 2


def test_bi_cross_validate_grouped_columns_and_missing_values():
    df = _make_synthetic_dataset(n_per_cluster=15, n_cols=12, seed=3)
    rng = np.random.default_rng(3)
    mask = rng.random(df.shape) < 0.2
    df_miss = df.mask(mask)
    col_groups = np.repeat(["exp1", "exp2", "exp3"], 4)

    res = bi_cross_validate(
        df_miss,
        n_components=2,
        seq_weight=[0.0, 10.0],
        row_folds=3,
        col_folds=3,
        col_groups=col_groups,
        train_col_fraction=0.0,
        random_state=0,
        n_epochs=6,
    )
    assert len(res) == 2
    assert np.all(np.isfinite(res[["log_likelihood", "mse", "r2"]]))
    # With J_train = empty (train_col_fraction=0.0), sequence motif weight w=10
    # must outperform w=0 (which has no signal to distinguish clusters).
    ll_w0 = float(res.loc[res["seq_weight"] == 0.0, "log_likelihood"].iloc[0])
    ll_w10 = float(res.loc[res["seq_weight"] == 10.0, "log_likelihood"].iloc[0])
    assert ll_w10 > ll_w0


def test_bi_cross_validate_rejects_bad_fraction_and_metric():
    df = _make_synthetic_dataset(n_per_cluster=10, n_cols=6, seed=4)
    with pytest.raises(ValueError, match="train_col_fraction"):
        bi_cross_validate(df, n_components=2, seq_weight=1.0, train_col_fraction=1.5)

    res = pd.DataFrame(
        {
            "n_components": [2],
            "seq_weight": [1.0],
            "log_likelihood": [-1.0],
            "mse": [0.5],
            "r2": [0.8],
            "n_obs": [60],
        }
    )
    with pytest.raises(ValueError, match="Unsupported metric"):
        select_best_hyperparams(res, metric="bogus")  # type: ignore[arg-type]
