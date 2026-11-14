"""Dual data and motif clustering (DDMC).

Contains the `DDMC` model itself — a `sklearn.mixture.GaussianMixture`
subclass that jointly clusters peptides on their phosphorylation signal and
their sequence motif — and `get_pspl_pssm_distances`, the helper it uses to
compare cluster motifs against kinase specificity profiles.
"""

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import kmeans_plusplus
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .binomial import (
    AAlist,
    BackgroundSeqs,
    Binomial,
    frequencies,
    normalize_ptm_sequences,
)
from .motifs import compute_control_pssm, get_pspls
from .pam250 import PAM250


class DDMC(GaussianMixture):
    """Cluster peptides by both sequence similarity and condition-wise phosphorylation following an
    expectation-maximization algorithm.

    `DDMC` subclasses `sklearn.mixture.GaussianMixture` and reuses its EM
    loop, but scores each peptide against each cluster using both the usual
    Gaussian mixture log-probability over its phosphorylation signal and a
    sequence-motif term (weighted by `seq_weight`), and refits both the
    Gaussian mixture parameters and the per-cluster sequence motif at every
    M step. See `ddmc.binomial.Binomial` and `ddmc.pam250.PAM250` for the
    two available motif models.

    Attributes set by `fit`:
        p_signal: The `p_signal` DataFrame passed to `fit`.
        sequences: `p_signal.index`, as an upper-cased numpy array.
        seq_dist: The fitted `Binomial` or `PAM250` sequence-distance model.
        scores_: Per-peptide, per-cluster responsibilities (soft cluster
            assignments) of shape (n_peptides, n_components).
        seq_scores_: Per-peptide, per-cluster weighted sequence
            log-probabilities (`seq_weight * seq_dist.logWeights`) from the
            last E step, of shape (n_peptides, n_components).
    """

    def __init__(
        self,
        n_components: int,
        seq_weight: float,
        distance_method: Literal["PAM250", "Binomial"] = "Binomial",
        random_state: int | np.random.RandomState | None = None,
        max_iter: int = 200,
        tol: float = 1e-4,
        n_epochs: int = 10,
        include_ptms: bool = True,
    ):
        """
        Args:
            n_components: The number of clusters to fit.
            seq_weight: Weight applied to the sequence-motif log-probability
                relative to the Gaussian mixture log-probability when
                scoring each peptide against each cluster. `0` reduces
                `DDMC` to an ordinary Gaussian mixture model.
            distance_method: Which sequence-distance model to use for the
                motif term: `"Binomial"` (`ddmc.binomial.Binomial`) or
                `"PAM250"` (`ddmc.pam250.PAM250`).
            random_state: Seed or `numpy.random.RandomState` controlling the
                random initialization and mini-batch shuffling, for
                reproducibility.
            max_iter: Maximum number of EM iterations (kept for API
                compatibility).
            tol: Convergence threshold (kept for API compatibility).
            n_epochs: Number of mini-batch EMA epochs to run before the final
                exact full-batch EM pass.
        """
        super().__init__(
            n_components=n_components,
            covariance_type="diag",
            n_init=1,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.distance_method = distance_method
        self.seq_weight = seq_weight
        self.n_epochs = n_epochs
        self.include_ptms = include_ptms

    def _gen_peptide_distances(self, sequences, distance_method) -> None:
        """Build `self.seq_dist`, the sequence-distance model used for the
        motif term of the E and M steps.

        Args:
            sequences: The length-11 peptide sequences being clustered.
            distance_method: Which sequence-distance model to construct:
                `"Binomial"` or `"PAM250"`.
        """
        sequences = normalize_ptm_sequences(sequences, include_ptms=self.include_ptms)
        self.sequences = sequences
        if distance_method == "PAM250":
            self.seq_dist: PAM250 | Binomial = PAM250(
                sequences, include_ptms=self.include_ptms
            )
        elif distance_method == "Binomial":
            self.seq_dist = Binomial(sequences, include_ptms=self.include_ptms)
        else:
            raise ValueError("Wrong distance type.")

    @staticmethod
    def _diagonal_gaussian_log_prob(
        X_zero: np.ndarray,
        obs_mask: np.ndarray,
        means: np.ndarray,
        precisions_cholesky: np.ndarray,
    ) -> np.ndarray:
        """Exact marginal diagonal Gaussian log-probability over observed features."""
        p2 = precisions_cholesky * precisions_cholesky
        log_det = np.log(2.0 * np.pi) - 2.0 * np.log(precisions_cholesky)
        const_kf = log_det + p2 * (means * means)
        linear_kf = 2.0 * p2 * means
        return -0.5 * (
            obs_mask @ const_kf.T - X_zero @ linear_kf.T + (X_zero * X_zero) @ p2.T
        )

    def _estimate_observed_gaussian_log_prob(
        self, X: np.ndarray, obs_mask: np.ndarray
    ) -> np.ndarray:
        """Exact marginal diagonal Gaussian log-probability over observed features."""
        assert self.means_ is not None and self.precisions_cholesky_ is not None
        return self._diagonal_gaussian_log_prob(
            X, obs_mask, self.means_, self.precisions_cholesky_
        )

    def _estimate_log_prob(self, X: np.ndarray, xp=None) -> np.ndarray:
        """EM E-step helper. Estimate the log-probability of each peptide
        under each cluster, combining the Gaussian mixture log-probability
        over `X` (marginalizing any missing entries) with the weighted
        sequence-motif log-probability.

        Args:
            X: Phosphorylation signal of shape (n_samples, n_features), with
                missing entries zeroed when `self._missing` is True.
            xp: Array-API namespace to use, forwarded to
                `GaussianMixture._estimate_log_prob` (unused directly here).

        Returns:
            Combined log-probability of each sample under each cluster, of
            shape (n_samples, n_components). Also stored as
            `self.seq_scores_` (the sequence-only term).
        """
        if self._missing:
            logp = self._estimate_observed_gaussian_log_prob(X, self._obs_mask)
        else:
            logp = super()._estimate_log_prob(X, xp=xp)

        # Add in the sequence effect
        self.seq_scores_ = self.seq_weight * self.seq_dist.logWeights
        logp += self.seq_scores_

        return logp

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray, xp=None) -> None:
        """EM M-step. Refit both the Gaussian mixture parameters (from observed
        features only when `self._missing` is True) and the sequence-motif
        model from the current responsibilities.

        Args:
            X: Phosphorylation signal of shape (n_samples, n_features), with
                missing entries zeroed when `self._missing` is True.
            log_resp: Logarithm of the posterior probabilities (or
                responsibilities) of each sample in `X`, of shape
                (n_samples, n_components).
            xp: Array-API namespace to use, forwarded to
                `GaussianMixture._m_step` (unused directly here).
        """
        resp = np.exp(log_resp)
        if self._missing:
            n_ema = np.sum(resp, axis=0)
            n_feat_ema = resp.T @ self._obs_mask
            s_ema = resp.T @ X
            q_ema = resp.T @ (X * X)
            self._update_gmm_params_from_ema(n_ema, n_feat_ema, s_ema, q_ema)
        else:
            super()._m_step(X, log_resp, xp=xp)  # Do the regular m step

        # Do sequence m step
        self.seq_dist.from_summaries(resp)

    @staticmethod
    def _emeans_schedule(n: int, k: int, n_epochs: int) -> tuple[int, float]:
        """Compute mini-batch size and initial EMA step size from `e-means` scaling laws."""
        t_budget = max(n_epochs, 1)
        raw_b = 0.035 * (k**0.13) * (n**0.88) * (t_budget**0.81)
        min_b = min(n, max(k, 16))
        batch_size = int(np.clip(np.round(raw_b), min_b, n))
        alpha_0 = float(
            min(
                0.824
                * (k**-0.46)
                * (batch_size**0.90)
                * (n**-0.45)
                * (t_budget**-0.11),
                1.0,
            )
        )
        return batch_size, alpha_0

    @staticmethod
    def _make_batches(
        n: int, batch_size: int, rng: np.random.RandomState
    ) -> list[np.ndarray]:
        """Shuffle indices into mini-batches, merging any small trailing remainder."""
        perm = rng.permutation(n)
        if batch_size >= n:
            return [perm]
        batches = [perm[i : i + batch_size] for i in range(0, n, batch_size)]
        if len(batches) > 1 and len(batches[-1]) < max(1, batch_size // 2):
            batches[-2] = np.concatenate([batches[-2], batches[-1]])
            batches.pop()
        return batches

    @staticmethod
    def _cosine_alpha(alpha_0: float, step: int, total_steps: int) -> float:
        """Cosine-annealed EMA step size from `alpha_0` down to `0.1 * alpha_0`."""
        if total_steps <= 1:
            return alpha_0
        progress = step / (total_steps - 1)
        return 0.1 * alpha_0 + 0.5 * (0.9 * alpha_0) * (1.0 + np.cos(np.pi * progress))

    def _revive_dead_clusters(
        self,
        n_ema: np.ndarray,
        s_ema: np.ndarray,
        q_ema: np.ndarray | None,
        min_cluster_size: float,
        rng: np.random.RandomState,
        revive_seq: bool = False,
        n_feat_ema: np.ndarray | None = None,
    ) -> None:
        """Revive clusters whose effective count fell below `min_cluster_size`
        by proportionally sampling an active cluster and splitting it in two.
        """
        dead = np.where(n_ema < min_cluster_size)[0]
        if dead.size == 0:
            return
        d = s_ema.shape[1]
        for j in dead:
            weights = np.maximum(n_ema - min_cluster_size, 0.0)
            total_w = float(np.sum(weights))
            if total_w > 0.0:
                j_star = int(rng.choice(len(n_ema), p=weights / total_w))
            else:
                j_star = int(np.argmax(n_ema))
            if j_star == j or n_ema[j_star] <= 0.0:
                continue

            nf_old = (
                np.maximum(n_feat_ema[j_star], 1e-12)
                if n_feat_ema is not None
                else float(n_ema[j_star])
            )
            mean_star = s_ema[j_star] / nf_old
            if q_ema is not None:
                var_star = np.maximum(
                    q_ema[j_star] / nf_old - mean_star * mean_star, self.reg_covar
                )
            else:
                var_star = np.ones(d, dtype=s_ema.dtype)

            delta = 0.05 * np.sqrt(var_star) * rng.normal(size=d)
            mean_j = mean_star + delta
            mean_j_star = mean_star - delta

            half_n = 0.5 * float(n_ema[j_star])
            n_ema[j] = half_n
            n_ema[j_star] = half_n
            if n_feat_ema is not None:
                half_nf = 0.5 * n_feat_ema[j_star]
                n_feat_ema[j] = half_nf
                n_feat_ema[j_star] = half_nf
            else:
                half_nf = half_n
            s_ema[j] = half_nf * mean_j
            s_ema[j_star] = half_nf * mean_j_star
            if q_ema is not None:
                q_ema[j] = half_nf * (var_star + mean_j * mean_j)
                q_ema[j_star] = half_nf * (var_star + mean_j_star * mean_j_star)
            if revive_seq:
                self.seq_dist.split_cluster(j_star, int(j))

    def _init_emeans_centroids(
        self, X: np.ndarray, min_cluster_size: float, rng: np.random.RandomState
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize cluster centroids using `e-means` (k-means++ seed on a
        subsample followed by fast mini-batch EMA k-means with split revival).
        """
        n, d = X.shape
        k = self.n_components
        sub_n = min(n, max(2000, 20 * k))
        if sub_n < n:
            sub_idx = rng.choice(n, size=sub_n, replace=False)
            X_sub = X[sub_idx]
            if self._missing:
                assert self._col_means is not None
                X_sub = np.where(
                    self._obs_mask[sub_idx] > 0, X_sub, self._col_means[None, :]
                )
        else:
            X_sub = X
            if self._missing:
                assert self._col_means is not None
                X_sub = np.where(self._obs_mask > 0, X_sub, self._col_means[None, :])
        centers, _ = kmeans_plusplus(X_sub, n_clusters=k, random_state=rng)
        centers = centers.astype(X.dtype, copy=True)

        t_init = 3
        batch_size, alpha_0 = self._emeans_schedule(n, k, t_init)
        all_batches = [self._make_batches(n, batch_size, rng) for _ in range(t_init)]
        total_steps = sum(len(b) for b in all_batches)

        n_ema = np.full(k, n / k, dtype=np.float64)
        n_feat_ema = (
            np.full((k, d), n / k, dtype=np.float64)
            if self._missing
            else n_ema[:, None].copy()
        )
        s_ema = n_feat_ema * centers

        step = 0
        for epoch_batches in all_batches:
            for batch_idx in epoch_batches:
                alpha = self._cosine_alpha(alpha_0, step, total_steps)
                step += 1

                X_b = X[batch_idx]
                c_sq = centers * centers
                if self._missing:
                    m_b = self._obs_mask[batch_idx]
                    dists = m_b @ c_sq.T - 2.0 * (X_b @ centers.T)
                else:
                    dists = np.sum(c_sq, axis=1)[None, :] - 2.0 * (X_b @ centers.T)
                labels = np.argmin(dists, axis=1)

                scale = n / len(batch_idx)
                n_b = np.bincount(labels, minlength=k).astype(np.float64) * scale
                s_b = np.zeros((k, d), dtype=np.float64)
                np.add.at(s_b, labels, X_b)
                s_b *= scale

                n_ema = (1.0 - alpha) * n_ema + alpha * n_b
                s_ema = (1.0 - alpha) * s_ema + alpha * s_b
                if self._missing:
                    nf_b = np.zeros((k, d), dtype=np.float64)
                    np.add.at(nf_b, labels, m_b)
                    nf_b *= scale
                    n_feat_ema = (1.0 - alpha) * n_feat_ema + alpha * nf_b
                else:
                    n_feat_ema = n_ema[:, None]

                self._revive_dead_clusters(
                    n_ema,
                    s_ema,
                    None,
                    min_cluster_size,
                    rng,
                    revive_seq=False,
                    n_feat_ema=n_feat_ema if self._missing else None,
                )
                if self._missing:
                    assert self._col_means is not None
                    centers = np.where(
                        n_feat_ema > 1e-6,
                        s_ema / np.maximum(n_feat_ema, 1e-12),
                        self._col_means[None, :],
                    )
                else:
                    centers = s_ema / np.maximum(n_ema[:, None], 1e-12)

        return centers, n_ema

    def _update_gmm_params_from_ema(
        self,
        n_ema: np.ndarray,
        n_feat_ema: np.ndarray,
        s_ema: np.ndarray,
        q_ema: np.ndarray,
    ) -> None:
        """Recover GMM `weights_`, `means_`, `covariances_`, and
        `precisions_cholesky_` from EMA sufficient statistics.
        """
        n_safe = np.maximum(n_ema, 1e-12)
        self.weights_ = n_safe / np.sum(n_safe)
        nf_safe = np.maximum(n_feat_ema, 1e-12)
        self.means_ = s_ema / nf_safe
        if self._missing and self._col_means is not None:
            self.means_ = np.where(
                n_feat_ema > 1e-6, self.means_, self._col_means[None, :]
            )
        self.covariances_ = np.maximum(
            q_ema / nf_safe - self.means_ * self.means_, self.reg_covar
        )
        if self._missing and self._col_vars is not None:
            self.covariances_ = np.where(
                n_feat_ema > 1e-6,
                self.covariances_,
                np.maximum(self._col_vars[None, :], self.reg_covar),
            )
        self.precisions_cholesky_ = 1.0 / np.sqrt(self.covariances_)

    def _fit_emeans(self, X: np.ndarray) -> None:
        """Fit DDMC using mini-batch EMA updates, scaling-law batch size and
        cosine-scheduled step size, proportional split revival, and one final
        exact full-batch EM pass.
        """
        n, d = X.shape
        k = self.n_components
        rng = check_random_state(self.random_state)
        min_cluster_size = (
            min(1.0, 0.25 * n / k) if n < 4 * k else max(1.0, 0.02 * n / k)
        )

        centers, n_init = self._init_emeans_centroids(X, min_cluster_size, rng)

        global_var = (
            np.maximum(self._col_vars, self.reg_covar)
            if self._missing and self._col_vars is not None
            else np.maximum(np.var(X, axis=0), self.reg_covar)
        )
        n_ema = np.maximum(n_init, min_cluster_size).astype(np.float64)
        n_feat_ema = (
            np.broadcast_to(n_ema[:, None], (k, d)).copy()
            if self._missing
            else n_ema[:, None]
        )
        s_ema = n_feat_ema * centers
        q_ema = n_feat_ema * (global_var[None, :] + centers * centers)
        self._update_gmm_params_from_ema(n_ema, n_feat_ema, s_ema, q_ema)

        use_seq = self.seq_weight != 0
        if use_seq:
            self.seq_dist.init_ema(k)

        batch_size, alpha_0 = self._emeans_schedule(n, k, self.n_epochs)
        all_batches = [
            self._make_batches(n, batch_size, rng) for _ in range(self.n_epochs)
        ]
        total_steps = sum(len(b) for b in all_batches)

        # Seed sequence EMA from the first mini-batch using initial Gaussian
        # responsibilities so the motif model is active from step 0.
        if use_seq and total_steps > 0:
            assert self.weights_ is not None
            seed_idx = all_batches[0][0]
            log_prob_seed = (
                self._estimate_observed_gaussian_log_prob(
                    X[seed_idx], self._obs_mask[seed_idx]
                )
                if self._missing
                else GaussianMixture._estimate_log_prob(self, X[seed_idx])
            ) + np.log(self.weights_)
            log_resp_seed = log_prob_seed - logsumexp(
                log_prob_seed, axis=1, keepdims=True
            )
            self.seq_dist.update_ema(
                seed_idx, np.exp(log_resp_seed), alpha=1.0, scale=n / len(seed_idx)
            )
            self.seq_dist.finalize_ema()

        step = 0
        for epoch_batches in all_batches:
            for batch_idx in epoch_batches:
                alpha = self._cosine_alpha(alpha_0, step, total_steps)
                step += 1

                assert self.weights_ is not None
                X_b = X[batch_idx]
                if self._missing:
                    m_b = self._obs_mask[batch_idx]
                    log_prob_b = self._estimate_observed_gaussian_log_prob(
                        X_b, m_b
                    ) + np.log(self.weights_)
                else:
                    log_prob_b = GaussianMixture._estimate_log_prob(self, X_b) + np.log(
                        self.weights_
                    )
                if use_seq:
                    log_prob_b = (
                        log_prob_b
                        + self.seq_weight
                        * self.seq_dist.estimate_log_prob_batch(batch_idx)
                    )
                log_resp_b = log_prob_b - logsumexp(log_prob_b, axis=1, keepdims=True)
                resp_b = np.exp(log_resp_b)

                scale = n / len(batch_idx)
                n_b = np.sum(resp_b, axis=0) * scale
                s_b = (resp_b.T @ X_b) * scale
                q_b = (resp_b.T @ (X_b * X_b)) * scale

                n_ema = (1.0 - alpha) * n_ema + alpha * n_b
                s_ema = (1.0 - alpha) * s_ema + alpha * s_b
                q_ema = (1.0 - alpha) * q_ema + alpha * q_b
                if self._missing:
                    nf_b = (resp_b.T @ m_b) * scale
                    n_feat_ema = (1.0 - alpha) * n_feat_ema + alpha * nf_b
                else:
                    n_feat_ema = n_ema[:, None]

                if use_seq:
                    self.seq_dist.update_ema(
                        batch_idx, resp_b, alpha=alpha, scale=scale
                    )

                self._revive_dead_clusters(
                    n_ema,
                    s_ema,
                    q_ema,
                    min_cluster_size,
                    rng,
                    revive_seq=use_seq,
                    n_feat_ema=n_feat_ema if self._missing else None,
                )
                if not self._missing:
                    n_feat_ema = n_ema[:, None]
                self._update_gmm_params_from_ema(n_ema, n_feat_ema, s_ema, q_ema)
                if use_seq:
                    self.seq_dist.finalize_ema()

        # Exact full-batch finalization pass over all peptides.
        if use_seq:
            self.seq_dist.compute_log_weights()
        _, log_resp = self._e_step(X)
        self._m_step(X, log_resp)
        self.lower_bound_, log_resp = self._e_step(X)
        self.scores_ = np.exp(log_resp)
        self.converged_ = True
        self.n_iter_ = self.n_epochs

    def fit(self, p_signal: pd.DataFrame) -> "DDMC":  # ty: ignore[invalid-method-override]
        """
        Compute EM clustering.

        Args:
            p_signal: Dataframe of shape (number of peptides, number of samples)
                containing the phosphorylation signal. `p_signal.index` contains
                the length-11 AA sequence of each peptide, containing the
                phosphoacceptor in the middle and five AAs flanking it.

        Returns:
            self, fit to `p_signal`.
        """
        assert isinstance(p_signal, pd.DataFrame), (
            "`p_signal` must be a pandas dataframe."
        )
        sequences = p_signal.index.values

        for i, seq in enumerate(sequences):
            assert isinstance(seq, str), (
                f"Sequence {seq} at index {i} is not a string. All sequences must be strings."
            )
            assert len(seq) == 11, (
                f"Sequence {seq} at index {i} is of length {len(seq)}. All sequences must be of length 11."
            )
            assert all([token.upper() in AAlist for token in seq]), (
                f"Sequence {seq} at index {i} contains invalid characters."
            )

        assert (
            p_signal.select_dtypes(include=[np.number]).shape[1] == p_signal.shape[1]
        ), "All values in `p_signal` should be numerical"

        self.p_signal = p_signal
        self._gen_peptide_distances(sequences, self.distance_method)

        if np.any(np.isnan(p_signal)):
            self._missing = True
            X_raw = p_signal.to_numpy(dtype=np.float64)
            obs_mask = ~np.isnan(X_raw)
            self._obs_mask = obs_mask.astype(np.float64)
            col_counts = np.sum(self._obs_mask, axis=0)
            X = np.where(obs_mask, X_raw, 0.0)
            col_means = np.divide(
                np.sum(X, axis=0),
                col_counts,
                out=np.zeros(X.shape[1]),
                where=col_counts > 0,
            )
            self._col_means: np.ndarray | None = col_means
            centered = np.where(obs_mask, X - col_means[None, :], 0.0)
            self._col_vars: np.ndarray | None = np.divide(
                np.sum(centered * centered, axis=0),
                col_counts,
                out=np.ones(X.shape[1]),
                where=col_counts > 0,
            )
        else:
            self._missing = False
            self._col_means = None
            self._col_vars = None
            X = p_signal

        X = validate_data(self, X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X)
        self._fit_emeans(X)

        assert np.all(np.isfinite(self.scores_))
        assert np.all(np.isfinite(self.seq_scores_))
        return self

    @overload
    def transform(self, as_df: Literal[False] = False) -> np.ndarray: ...
    @overload
    def transform(self, as_df: Literal[True]) -> pd.DataFrame: ...
    def transform(self, as_df: bool = False) -> np.ndarray | pd.DataFrame:
        """
        Return cluster centers.

        Args:
            as_df: Whether or not the result should be wrapped in a dataframe with labeled axes.

        Returns:
            The cluster centers, either a np array or pd df of shape (n_samples, n_components).
        """
        check_is_fitted(self, ["means_"])
        assert self.means_ is not None
        centers = self.means_.T
        if as_df:
            centers = pd.DataFrame(
                centers,
                index=self.p_signal.columns,
                columns=np.arange(self.n_components),
            )
        return centers

    def impute(self) -> pd.DataFrame:
        """
        Imputes missing values in the dataset passed in fit() and returns the
        imputed dataset.

        Returns:
            A copy of the `p_signal` passed to `fit`, with each peptide's
            missing samples filled in from its assigned cluster's center.
        """
        p_signal = self.p_signal.copy()
        labels = self.labels()  # cluster assignments
        centers = self.transform()  # samples x clusters
        for ii in range(p_signal.shape[0]):
            p_signal.iloc[ii, np.isnan(p_signal.iloc[ii, :])] = centers[
                np.isnan(p_signal.iloc[ii, :]), labels[ii]
            ]
        assert np.all(np.isfinite(p_signal))
        return p_signal

    @overload
    def get_pssms(
        self, PsP_background: bool = False, clusters: None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...
    @overload
    def get_pssms(
        self, PsP_background: bool = False, *, clusters: list[int]
    ) -> np.ndarray: ...
    def get_pssms(
        self, PsP_background: bool = False, clusters: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Compute position-specific scoring matrix of each cluster.
        Note, to normalize by amino acid frequency this uses either
        all the sequences in the data set or a collection of random MS phosphosites in PhosphoSitePlus.

        Args:
            PsP_background: Whether or not PhosphoSitePlus should be used for background frequency.
            clusters: cluster indices to get pssms for

        Returns:
            If the clusters argument is used, an array of shape (len(clusters), 20, 11),
            else two arrays, where the first (of shape (n_pssms,))
            contains the clusters of the pssms in the second
            (of shape (n_pssms, 20, 11)).
        """
        pssm_names, pssms = [], []
        if PsP_background:
            bg_seqs = BackgroundSeqs(self.sequences)
            back_pssm = compute_control_pssm(bg_seqs)
        else:
            back_pssm = np.zeros((len(AAlist), 11), dtype=float)
            for seq in self.sequences:
                seq = seq.upper()
                for kk, aa in enumerate(seq):
                    back_pssm[AAlist.index(aa), kk] += 1.0
            for pos in range(back_pssm.shape[1]):
                if pos == 5:
                    continue
                back_pssm[:, pos] /= np.mean(back_pssm[:, pos])

        l1 = list(np.arange(self.n_components))
        l2 = list(set(self.labels()))
        ec = [i for i in l1 + l2 if i not in l1 or i not in l2]
        for ii in range(self.n_components):
            # Check for empty clusters and ignore them, if there are
            if ii in ec:
                continue

            # Compute PSSM
            pssm = np.zeros((len(AAlist), 11), dtype=float)
            for jj, seq in enumerate(self.sequences):
                seq = seq.upper()
                for kk, aa in enumerate(seq):
                    pssm[AAlist.index(aa), kk] += self.scores_[jj, ii]

            # Normalize by position across residues
            for pos in range(pssm.shape[1]):
                if pos == 5:
                    continue
                pssm[:, pos] /= np.mean(pssm[:, pos])

            # Normalize to background PSSM to account for AA frequencies per position
            old_settings = np.seterr(divide="ignore", invalid="ignore")
            pssm /= back_pssm.copy()
            np.seterr(**old_settings)

            # Log2 transform
            pssm = np.ma.log2(pssm)
            pssm = pssm.filled(0)
            pssm = np.nan_to_num(pssm)
            pssm = pd.DataFrame(pssm)
            pssm.index = AAlist

            # Normalize phosphoacceptor position to frequency
            df = pd.DataFrame({"Sequence": self.sequences})
            df["Cluster"] = self.labels()
            clSeq = df[df["Cluster"] == ii]["Sequence"]
            clSeq = pd.DataFrame(
                frequencies(np.char.upper(clSeq.to_numpy(dtype=str)).tolist())
            ).T
            tm = np.mean([clSeq.loc["S", 5], clSeq.loc["T", 5], clSeq.loc["Y", 5]])
            for p_site in ["S", "T", "Y"]:
                pssm.loc[p_site, 5] = np.log2(clSeq.loc[p_site, 5] / tm)

            pssms.append(np.clip(pssm, a_min=0, a_max=3))
            pssm_names.append(ii)

        pssm_names, pssms = np.array(pssm_names), np.array(pssms)

        if clusters is not None:
            return pssms[
                [np.where(pssm_names == cluster)[0][0] for cluster in clusters]
            ]

        return pssm_names, pssms

    def predict_upstream_kinases(
        self,
        PsP_background: bool = True,
    ) -> pd.DataFrame:
        """Compute matrix-matrix similarity between kinase specificity profiles
        and cluster PSSMs to identify upstream kinases regulating clusters.

        Args:
            PsP_background: Whether or not PhosphoSitePlus should be used
                for the background amino acid frequency when building each
                cluster's PSSM (see `get_pssms`).

        Returns:
            DataFrame of shape (n_kinases, n_nonempty_clusters) with a
            Frobenius distance between each kinase's specificity profile and
            each cluster's PSSM; smaller values indicate a better match.
        """
        kinases, pspls = get_pspls()
        clusters, pssms = self.get_pssms(PsP_background=PsP_background)
        distances = get_pspl_pssm_distances(
            pspls,
            pssms,
            as_df=True,
            pssm_names=clusters,
            kinases=kinases,
        )
        return distances

    def get_nonempty_clusters(self) -> np.ndarray:
        """List the clusters that at least one peptide is assigned to.

        Returns:
            Sorted array of the distinct cluster indices present in
            `self.labels()`; shorter than `n_components` if any clusters
            are empty.
        """
        return np.unique(self.labels())

    def has_empty_clusters(self) -> bool:
        """
        Checks whether the most recent call to fit() resulted in empty clusters.

        Returns:
            True if any of the `n_components` clusters has no peptides
            assigned to it.
        """
        check_is_fitted(self, ["scores_"])
        return self.get_nonempty_clusters().size != self.n_components

    def fold_in(
        self,
        p_signal_test: pd.DataFrame,
        cols: np.ndarray | Sequence[int] | None = None,
    ) -> np.ndarray:
        """Compute out-of-sample posterior cluster responsibilities for unseen
        peptides using only the specified sample columns `cols` and the
        fitted sequence-motif model.

        Args:
            p_signal_test: DataFrame of unseen peptides with length-11 sequences
                as index and the same sample columns as `self.p_signal`.
            cols: Integer column indices to condition on. If `None`, all
                columns are used; if empty, responsibilities are computed from
                cluster priors and sequence motifs alone.

        Returns:
            Posterior cluster responsibilities of shape
            `(len(p_signal_test), n_components)`.
        """
        check_is_fitted(self, ["means_", "precisions_cholesky_", "weights_"])
        assert (
            self.means_ is not None
            and self.precisions_cholesky_ is not None
            and self.weights_ is not None
        )
        col_idx = (
            np.arange(p_signal_test.shape[1], dtype=int)
            if cols is None
            else np.asarray(cols, dtype=int)
        )
        log_prob = np.broadcast_to(
            np.log(self.weights_), (len(p_signal_test), self.n_components)
        ).copy()

        if col_idx.size > 0:
            X_raw = p_signal_test.to_numpy(dtype=np.float64)[:, col_idx]
            obs_mask = (~np.isnan(X_raw)).astype(np.float64)
            X_zero = np.where(obs_mask > 0, X_raw, 0.0)
            log_prob += self._diagonal_gaussian_log_prob(
                X_zero,
                obs_mask,
                self.means_[:, col_idx],
                self.precisions_cholesky_[:, col_idx],
            )

        if self.seq_weight != 0:
            log_prob += self.seq_weight * self.seq_dist.score_sequences(
                np.asarray(p_signal_test.index.values, dtype=str)
            )

        log_resp = log_prob - logsumexp(log_prob, axis=1, keepdims=True)
        return np.exp(log_resp)

    def score_held_out(
        self,
        p_signal_test: pd.DataFrame,
        train_cols: np.ndarray | Sequence[int],
        test_cols: np.ndarray | Sequence[int],
    ) -> tuple[float, float, int, float, float]:
        """Evaluate bi-cross-validation metrics on held-out entries
        `(I_test, J_test)` after folding in `p_signal_test` using `train_cols`.

        Returns:
            Tuple `(ll_sum, sq_err_sum, n_obs, sum_y, sum_y2)` aggregated over
            all observed entries in `p_signal_test.iloc[:, test_cols]`.
        """
        check_is_fitted(self, ["means_", "precisions_cholesky_", "weights_"])
        assert self.means_ is not None and self.precisions_cholesky_ is not None

        gamma = self.fold_in(p_signal_test, cols=train_cols)
        test_idx = np.asarray(test_cols, dtype=int)
        X_te_raw = p_signal_test.to_numpy(dtype=np.float64)[:, test_idx]
        obs_te = ~np.isnan(X_te_raw)
        n_obs = int(np.sum(obs_te))
        if n_obs == 0:
            return 0.0, 0.0, 0, 0.0, 0.0

        X_te_zero = np.where(obs_te, X_te_raw, 0.0)
        log_prob_te = self._diagonal_gaussian_log_prob(
            X_te_zero,
            obs_te.astype(np.float64),
            self.means_[:, test_idx],
            self.precisions_cholesky_[:, test_idx],
        )
        has_obs = np.any(obs_te, axis=1)
        log_gamma = np.log(np.maximum(gamma[has_obs], 1e-300))
        ll_sum = float(np.sum(logsumexp(log_gamma + log_prob_te[has_obs], axis=1)))

        pred = gamma @ self.means_[:, test_idx]
        diff = X_te_raw[obs_te] - pred[obs_te]
        sq_err_sum = float(np.sum(diff * diff))
        y_obs = X_te_raw[obs_te]
        sum_y = float(np.sum(y_obs))
        sum_y2 = float(np.sum(y_obs * y_obs))
        return ll_sum, sq_err_sum, n_obs, sum_y, sum_y2

    def predict(self) -> np.ndarray:  # ty: ignore[invalid-method-override]
        """Provided the current model parameters, predict the cluster each peptide belongs to.

        Returns:
            Array of shape (n_peptides,) giving the index of the
            highest-likelihood cluster for each peptide in `self.p_signal`.
        """
        check_is_fitted(self, ["scores_"])
        return np.argmax(self.scores_, axis=1)

    def labels(self) -> np.ndarray:
        """Find cluster assignment with highest likelihood for each peptide.

        Returns:
            Array of shape (n_peptides,) giving each peptide's cluster
            index. Equivalent to `predict()`.
        """
        return self.predict()

    def score(self) -> float:  # ty: ignore[invalid-method-override]
        """Generate score of the fitting.

        Returns:
            The lower bound on the log-likelihood of the fitted model
            (`self.lower_bound_`, set by `GaussianMixture.fit`).
        """
        check_is_fitted(self, ["lower_bound_"])
        return self.lower_bound_


@overload
def get_pspl_pssm_distances(
    pspls: np.ndarray,
    pssms: np.ndarray,
    as_df: Literal[False] = False,
    pssm_names: Sequence | np.ndarray | None = None,
    kinases: Sequence | np.ndarray | None = None,
) -> np.ndarray: ...
@overload
def get_pspl_pssm_distances(
    pspls: np.ndarray,
    pssms: np.ndarray,
    as_df: Literal[True],
    pssm_names: Sequence | np.ndarray | None = None,
    kinases: Sequence | np.ndarray | None = None,
) -> pd.DataFrame: ...
def get_pspl_pssm_distances(
    pspls: np.ndarray,
    pssms: np.ndarray,
    as_df: bool = False,
    pssm_names: Sequence | np.ndarray | None = None,
    kinases: Sequence | np.ndarray | None = None,
) -> np.ndarray | pd.DataFrame:
    """
    Computes a distance matrix between PSPLs and PSSMs.

    Args:
        pspls: kinase specificity profiles of shape (n_kinase, 20, 9)
        pssms: position-specific scoring matrices of shape (n_pssms, 20, 11)
        as_df: Whether or not the returned matrix should be returned as a
            dataframe. Requires pssm_names and kinases.
        pssm_names: list of names for the pssms of shape (n_pssms,)
        kinases: list of names for the pspls of shape (n_kinase,)

    Returns:
        Distance matrix of shape (n_kinase, n_pssms).
    """
    assert pssms.shape[1:3] == (20, 11)
    assert pspls.shape[1:3] == (20, 9)
    pssms = np.delete(pssms, [5, 10], axis=2)
    dists = np.linalg.norm(pspls[:, None, :, :] - pssms[None, :, :, :], axis=(2, 3))
    if as_df:
        kinases = list(kinases) if kinases is not None else None
        pssm_names = list(pssm_names) if pssm_names is not None else None
        dists = pd.DataFrame(dists, index=kinases, columns=pssm_names)
    return dists
