import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from ddmc.clustering import DDMC
from ddmc.datasets import simulate_dropout


def _make_sparse_motif_dataset(
    n_per_cluster: int = 25,
    n_samples: int = 12,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a synthetic 3-module phosphoproteomics dataset with distinct motifs."""
    rng = np.random.default_rng(seed)
    seqs = (
        ["AAAPAsPAAAA"] * n_per_cluster
        + ["AARRRsAAAAA"] * n_per_cluster
        + ["AAAAAyEAAAA"] * n_per_cluster
    )
    true_labels = np.repeat([0, 1, 2], n_per_cluster)

    # Distinct spatial/sample profiles per kinase module
    means = np.array(
        [
            [2.5] * 4 + [-1.5] * 4 + [0.0] * (n_samples - 8),
            [-2.0] * 4 + [2.5] * 4 + [-0.5] * (n_samples - 8),
            [0.0] * 4 + [-1.0] * 4 + [3.0] * (n_samples - 8),
        ],
        dtype=np.float64,
    )
    X = np.vstack(
        [
            means[c][None, :] + rng.normal(0.0, 0.4, size=(n_per_cluster, n_samples))
            for c in range(3)
        ]
    )
    cols = [f"roi_{j}" for j in range(n_samples)]
    return pd.DataFrame(X, index=seqs, columns=cols), true_labels


def test_impute_soft_vs_hard_and_posterior_variance():
    rng = np.random.default_rng(0)
    seqs = ["AAAAAsAAAAA"] * 12 + ["RRRRRsRRRRR"] * 12
    data = np.vstack(
        [
            rng.normal(loc=2.5, scale=2.0, size=(12, 4)),
            rng.normal(loc=-2.5, scale=2.0, size=(12, 4)),
        ]
    )
    # Peptide 0 is clearly in cluster 0; peptide 24 has neutral sequence and 0.0
    # in observed cols so it has intermediate responsibility across both clusters.
    data[0, :] = [5.0, 5.0, 5.0, np.nan]
    ambiguous_row = np.array([[0.0, 0.0, np.nan, np.nan]])
    data = np.vstack([data, ambiguous_row])
    seqs = seqs + ["GGGGGsGGGGG"]

    df = pd.DataFrame(data, index=seqs, columns=["s1", "s2", "s3", "s4"])
    model = DDMC(n_components=2, seq_weight=1.0, random_state=0).fit(df)

    soft_df, soft_var = model.impute(mode="soft", return_var=True)
    hard_df, hard_var = model.impute(mode="hard", return_var=True)

    assert soft_df.shape == df.shape
    assert soft_var.shape == df.shape
    assert np.all(np.isfinite(soft_df))
    assert np.all(np.isfinite(soft_var))

    # Observed entries must be unchanged and have 0 posterior variance
    obs_mask = ~np.isnan(df.to_numpy())
    np.testing.assert_allclose(soft_df.to_numpy()[obs_mask], df.to_numpy()[obs_mask])
    np.testing.assert_allclose(soft_var.to_numpy()[obs_mask], 0.0)
    np.testing.assert_allclose(hard_var.to_numpy()[obs_mask], 0.0)

    # Exact formula check for soft expectation & law-of-total-variance
    gamma = model.scores_
    means = model.means_
    covs = model.covariances_
    assert means is not None and covs is not None

    expected_soft = gamma @ means
    expected_var = np.maximum(
        gamma @ (covs + means * means) - expected_soft * expected_soft, 0.0
    )
    assert soft_df.iloc[-1, 2] == pytest.approx(expected_soft[-1, 2])
    assert soft_var.iloc[-1, 2] == pytest.approx(expected_var[-1, 2])

    # Ambiguous peptide should have different soft vs hard imputation and
    # strictly larger posterior variance than confident peptide 0
    assert abs(soft_df.iloc[-1, 2] - hard_df.iloc[-1, 2]) > 0.1
    assert soft_var.iloc[-1, 2] > soft_var.iloc[0, 3]

    # Hard mode variance equals assigned cluster variance
    labels = model.labels()
    assert hard_var.iloc[0, 3] == pytest.approx(covs[labels[0], 3])

    with pytest.raises(ValueError, match="Unknown imputation mode"):
        model.impute(mode="invalid")  # type: ignore[arg-type]


def test_simulate_dropout_mcar_mnar_block():
    rng = np.random.default_rng(1)
    X = rng.normal(loc=0.0, scale=2.0, size=(30, 8))
    X[0, 0] = np.nan  # pre-existing missing entry
    df = pd.DataFrame(X, columns=[f"c{j}" for j in range(8)])

    # MCAR
    mcar = simulate_dropout(df, missing_fraction=0.5, mechanism="mcar", random_state=0)
    assert np.isnan(mcar.iloc[0, 0])
    n_obs_orig = int(np.sum(~np.isnan(df.to_numpy())))
    n_obs_mcar = int(np.sum(~np.isnan(mcar.to_numpy())))
    assert n_obs_mcar == n_obs_orig - int(round(0.5 * n_obs_orig))

    # MNAR: masked values should have lower mean intensity than retained values
    mnar = simulate_dropout(df, missing_fraction=0.4, mechanism="mnar", random_state=0)
    newly_masked = (~np.isnan(df.to_numpy())) & np.isnan(mnar.to_numpy())
    retained = ~np.isnan(mnar.to_numpy())
    assert float(np.mean(df.to_numpy()[newly_masked])) < float(
        np.mean(df.to_numpy()[retained])
    )

    # Block dropout: entire col_groups blocks should be either all-observed or all-NaN
    # (for rows without pre-existing NaNs)
    col_groups = ["expA", "expA", "expB", "expB", "expC", "expC", "expD", "expD"]
    block = simulate_dropout(
        df.iloc[1:],
        missing_fraction=0.5,
        mechanism="block",
        col_groups=col_groups,
        random_state=0,
    )
    block_vals = block.to_numpy()
    for r in range(block_vals.shape[0]):
        for g_start in range(0, 8, 2):
            pair_nans = np.isnan(block_vals[r, g_start : g_start + 2])
            assert np.all(pair_nans) or not np.any(pair_nans)

    with pytest.raises(ValueError, match="missing_fraction"):
        simulate_dropout(df, missing_fraction=1.0)
    with pytest.raises(ValueError, match="Unknown dropout mechanism"):
        simulate_dropout(df, missing_fraction=0.2, mechanism="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="col_groups"):
        simulate_dropout(df, missing_fraction=0.2, mechanism="block", col_groups=[0, 1])


def test_observed_only_marginalization_and_sequence_prior_under_75pct_dropout():
    full_df, true_labels = _make_sparse_motif_dataset(
        n_per_cluster=30, n_samples=12, seed=42
    )
    sparse_df = simulate_dropout(
        full_df, missing_fraction=0.75, mechanism="mcar", random_state=42
    )

    # Fit DDMC with sequence prior vs expression-only (seq_weight=0)
    model_prior = DDMC(
        n_components=3, seq_weight=5.0, distance_method="Binomial", random_state=0
    ).fit(sparse_df)
    model_noprior = DDMC(
        n_components=3, seq_weight=0.0, distance_method="Binomial", random_state=0
    ).fit(sparse_df)

    # Cluster recovery vs ground truth under 75% missingness
    ari_prior = adjusted_rand_score(true_labels, model_prior.labels())
    ari_noprior = adjusted_rand_score(true_labels, model_noprior.labels())
    assert ari_prior > 0.85
    assert ari_prior > ari_noprior

    # Held-out reconstruction MSE on the 75% masked entries
    masked = np.isnan(sparse_df.to_numpy())
    truth = full_df.to_numpy()[masked]

    imp_prior = model_prior.impute(mode="soft").to_numpy()[masked]
    imp_noprior = model_noprior.impute(mode="soft").to_numpy()[masked]
    col_means = np.nanmean(sparse_df.to_numpy(), axis=0)
    imp_colmean = np.broadcast_to(col_means, full_df.shape)[masked]

    mse_prior = float(np.mean((truth - imp_prior) ** 2))
    mse_noprior = float(np.mean((truth - imp_noprior) ** 2))
    mse_colmean = float(np.mean((truth - imp_colmean) ** 2))

    assert mse_prior < mse_noprior
    assert mse_prior < mse_colmean
