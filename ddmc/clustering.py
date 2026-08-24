"""Dual data and motif clustering (DDMC).

Contains the `DDMC` model itself — a `sklearn.mixture.GaussianMixture`
subclass that jointly clusters peptides on their phosphorylation signal and
their sequence motif — and `get_pspl_pssm_distances`, the helper it uses to
compare cluster motifs against kinase specificity profiles.
"""

import warnings
from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import pandas as pd
from fancyimpute import SoftImpute
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted

from .binomial import AAlist, BackgroundSeqs, Binomial, frequencies
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
                random initialization of the underlying Gaussian mixture,
                for reproducibility.
            max_iter: Maximum number of EM iterations to run.
            tol: Convergence threshold on the change in per-sample average
                log-likelihood between EM iterations.
        """
        super().__init__(
            n_components=n_components,
            covariance_type="diag",
            n_init=2,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.distance_method = distance_method
        self.seq_weight = seq_weight

    def _gen_peptide_distances(self, sequences, distance_method) -> None:
        """Build `self.seq_dist`, the sequence-distance model used for the
        motif term of the E and M steps.

        Args:
            sequences: The length-11 peptide sequences being clustered.
            distance_method: Which sequence-distance model to construct:
                `"Binomial"` or `"PAM250"`.
        """
        sequences = np.asarray(sequences, dtype=str)
        sequences = np.char.upper(sequences)
        self.sequences = sequences
        if distance_method == "PAM250":
            self.seq_dist: PAM250 | Binomial = PAM250(sequences)
        elif distance_method == "Binomial":
            self.seq_dist = Binomial(sequences)
        else:
            raise ValueError("Wrong distance type.")

    def _estimate_log_prob(self, X: np.ndarray, xp=None) -> np.ndarray:
        """EM E-step helper. Estimate the log-probability of each peptide
        under each cluster, combining the Gaussian mixture log-probability
        over `X` with the weighted sequence-motif log-probability.

        Args:
            X: Phosphorylation signal of shape (n_samples, n_features), with
                any missing values already imputed.
            xp: Array-API namespace to use, forwarded to
                `GaussianMixture._estimate_log_prob` (unused directly here).

        Returns:
            Combined log-probability of each sample under each cluster, of
            shape (n_samples, n_components). Also stored as
            `self.seq_scores_` (the sequence-only term).
        """
        logp = super()._estimate_log_prob(X, xp=xp)  # Do the regular work

        # Add in the sequence effect
        self.seq_scores_ = self.seq_weight * self.seq_dist.logWeights
        logp += self.seq_scores_

        return logp

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray, xp=None) -> None:
        """EM M-step. Impute missing values from the current cluster
        centers, then refit both the Gaussian mixture parameters and the
        sequence-motif model from the current responsibilities.

        Args:
            X: Phosphorylation signal of shape (n_samples, n_features). If
                `self._missing`, entries at `self.missing_d` are overwritten
                in place with each peptide's assigned cluster's center
                before the regular Gaussian mixture M step runs.
            log_resp: Logarithm of the posterior probabilities (or
                responsibilities) of each sample in `X`, of shape
                (n_samples, n_components).
            xp: Array-API namespace to use, forwarded to
                `GaussianMixture._m_step` (unused directly here).
        """
        if self._missing:
            labels = np.argmax(log_resp, axis=1)
            centers = np.array(self.means_)  # samples x clusters
            centers_fill = centers[labels, :]

            assert centers_fill.shape == X.shape
            X[self.missing_d] = centers_fill[self.missing_d]

        super()._m_step(X, log_resp, xp=xp)  # Do the regular m step

        # Do sequence m step
        self.seq_dist.from_summaries(np.exp(log_resp))

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
            self.missing_d = np.isnan(p_signal)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_signal = SoftImpute(verbose=False).fit_transform(p_signal)
        else:
            self._missing = False

        super().fit(p_signal)
        self.scores_ = self.predict_proba(p_signal)

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
                np.isnan(p_signal.iloc[ii, :]), labels[ii] - 1
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
                    pssm[AAlist.index(aa), kk] += self.scores_[jj, ii - 1]
                    if ii == 1 and not PsP_background:
                        back_pssm[AAlist.index(aa), kk] += 1.0

            # Normalize by position across residues
            for pos in range(pssm.shape[1]):
                if pos == 5:
                    continue
                pssm[:, pos] /= np.mean(pssm[:, pos])
                if ii == 1 and not PsP_background:
                    back_pssm[:, pos] /= np.mean(back_pssm[:, pos])

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
            clSeq = pd.DataFrame(frequencies(clSeq)).T
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
