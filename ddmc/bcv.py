"""Bi-cross-validation (BCV) for joint hyperparameter selection in `DDMC`.

Implements Owen & Perry (2009) bi-cross-validation adapted to joint
phosphorylation-signal and sequence-motif mixture models (`n_components`
and `seq_weight`), supporting grouped column folds (e.g. TMT experiments),
missing-data marginalization, and `train_col_fraction` sweeps down to
sequence-only fold-in.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.utils import check_random_state

from .clustering import DDMC


def _as_list(value: int | float | Sequence[int] | Sequence[float]) -> list:
    if isinstance(value, Sequence | np.ndarray) and not isinstance(value, str | bytes):
        return list(value)
    return [value]


def bi_cross_validate(
    p_signal: pd.DataFrame,
    n_components: int | Sequence[int],
    seq_weight: float | Sequence[float],
    distance_method: Literal["PAM250", "Binomial"] = "Binomial",
    row_folds: int = 5,
    col_folds: int = 5,
    col_groups: Sequence | np.ndarray | None = None,
    train_col_fraction: float | None = None,
    random_state: int | np.random.RandomState | None = None,
    n_epochs: int = 10,
) -> pd.DataFrame:
    """Evaluate `(n_components, seq_weight)` grids via leak-free bi-cross-validation.

    Partitions both peptides (rows) into `row_folds` folds and samples (columns)
    into `col_folds` folds (optionally grouped by `col_groups`, such as TMT
    experiments). For each row fold and `(k, w)` pair, a `DDMC` model is fit
    once on the training peptides `I_train`, then evaluated on every held-out
    quadrant `(I_test, J_test)` by folding in `I_test` using only `J_train` and
    the peptide sequences `S_test`.

    Args:
        p_signal: Phosphorylation signal DataFrame of shape
            `(n_peptides, n_samples)` with length-11 sequences as index.
        n_components: Cluster count(s) `k` to evaluate.
        seq_weight: Sequence motif weight(s) `w` to evaluate.
        distance_method: Sequence model (`"Binomial"` or `"PAM250"`).
        row_folds: Number of peptide (row) CV folds.
        col_folds: Number of sample (column) CV folds.
        col_groups: Optional group labels of length `p_signal.shape[1]` (e.g.,
            `CPTAC().get_sample_to_experiment()`) so correlated channels from
            the same MS experiment are never split across `J_train` and `J_test`.
        train_col_fraction: Optional fraction in `[0.0, 1.0]` of `J_train`
            columns to retain during fold-in. `0.0` evaluates sequence-only
            prediction (`J_train = empty`).
        random_state: Seed or `RandomState` for fold splitting and `DDMC`.
        n_epochs: Number of `e-means` mini-batch EMA epochs per `DDMC.fit`.

    Returns:
        DataFrame with one row per `(n_components, seq_weight)` pair and columns
        `["n_components", "seq_weight", "log_likelihood", "mse", "r2", "n_obs"]`.
    """
    if train_col_fraction is not None and not (0.0 <= train_col_fraction <= 1.0):
        raise ValueError("`train_col_fraction` must be between 0.0 and 1.0.")

    k_values = [int(k) for k in _as_list(n_components)]
    w_values = [float(w) for w in _as_list(seq_weight)]
    rng = check_random_state(random_state)

    row_cv = KFold(n_splits=row_folds, shuffle=True, random_state=rng)
    row_splits = list(row_cv.split(np.arange(p_signal.shape[0])))

    col_indices = np.arange(p_signal.shape[1])
    if col_groups is not None:
        col_splits = list(
            GroupKFold(n_splits=col_folds).split(col_indices, groups=col_groups)
        )
    else:
        col_cv = KFold(n_splits=col_folds, shuffle=True, random_state=rng)
        col_splits = list(col_cv.split(col_indices))

    # Pre-sample J_train subsets when train_col_fraction is specified so every
    # (k, w) pair is evaluated on the exact same held-in column subsets.
    fold_col_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for train_cols, test_cols in col_splits:
        if train_col_fraction is None:
            fold_train_cols = train_cols
        elif train_col_fraction == 0.0:
            fold_train_cols = np.empty(0, dtype=int)
        else:
            n_keep = max(1, int(round(train_col_fraction * len(train_cols))))
            fold_train_cols = rng.choice(train_cols, size=n_keep, replace=False)
        fold_col_pairs.append((fold_train_cols, test_cols))

    records: list[dict[str, float | int]] = []
    for k in k_values:
        for w in w_values:
            total_ll = 0.0
            total_sq_err = 0.0
            total_obs = 0
            total_y = 0.0
            total_y2 = 0.0

            for train_rows, test_rows in row_splits:
                p_train = p_signal.iloc[train_rows]
                p_test = p_signal.iloc[test_rows]
                seed = None if random_state is None else int(rng.randint(0, 2**31 - 1))
                model = DDMC(
                    n_components=k,
                    seq_weight=w,
                    distance_method=distance_method,
                    random_state=seed,
                    n_epochs=n_epochs,
                ).fit(p_train)

                for train_cols, test_cols in fold_col_pairs:
                    ll_sum, sq_err, n_obs, sum_y, sum_y2 = model.score_held_out(
                        p_test, train_cols, test_cols
                    )
                    total_ll += ll_sum
                    total_sq_err += sq_err
                    total_obs += n_obs
                    total_y += sum_y
                    total_y2 += sum_y2

            if total_obs == 0:
                ll_mean = float("nan")
                mse = float("nan")
                r2 = float("nan")
            else:
                ll_mean = total_ll / total_obs
                mse = total_sq_err / total_obs
                total_ss = total_y2 - (total_y * total_y) / total_obs
                r2 = 1.0 - total_sq_err / total_ss if total_ss > 0.0 else float("nan")

            records.append(
                {
                    "n_components": k,
                    "seq_weight": w,
                    "log_likelihood": ll_mean,
                    "mse": mse,
                    "r2": r2,
                    "n_obs": total_obs,
                }
            )

    return pd.DataFrame.from_records(records)


def select_best_hyperparams(
    bcv_results: pd.DataFrame,
    metric: Literal["log_likelihood", "mse", "r2"] = "log_likelihood",
) -> tuple[int, float]:
    """Return the `(n_components, seq_weight)` pair that optimizes `metric`."""
    if metric in ("log_likelihood", "r2"):
        best_idx = bcv_results[metric].idxmax()
    elif metric == "mse":
        best_idx = bcv_results[metric].idxmin()
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")
    row = bcv_results.loc[best_idx]
    return int(row["n_components"]), float(row["seq_weight"])
