""" Clustering functions. """

from typing import Literal, List, Sequence, Tuple
import warnings
from copy import deepcopy
import itertools
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted
from .binomial import Binomial, AAlist, BackgroundSeqs, frequencies
from .pam250 import PAM250
from .motifs import get_pspls, compute_control_pssm
from fancyimpute import SoftImpute


class DDMC(GaussianMixture):
    """Cluster peptides by both sequence similarity and condition-wise phosphorylation following an
    expectation-maximization algorithm."""

    def __init__(
        self,
        n_components: int,
        seq_weight: float,
        distance_method: Literal["PAM250", "Binomial"] = "Binomial",
        random_state=None,
        max_iter=200,
        tol=1e-4,
    ):
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

    def _gen_peptide_distances(self, sequences: np.ndarray, distance_method):
        if sequences.dtype != str:
            sequences = sequences.astype("str")
        sequences = np.char.upper(sequences)
        self.sequences = sequences
        if distance_method == "PAM250":
            self.seq_dist: PAM250 | Binomial = PAM250(sequences)
        elif distance_method == "Binomial":
            self.seq_dist = Binomial(sequences)
        else:
            raise ValueError("Wrong distance type.")

    def _estimate_log_prob(self, X: np.ndarray):
        """Estimate the log-probability of each point in each cluster."""
        logp = super()._estimate_log_prob(X)  # Do the regular work

        # Add in the sequence effect
        self.seq_scores_ = self.seq_weight * self.seq_dist.logWeights
        logp += self.seq_scores_

        return logp

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray):
        """M step.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        if self._missing:
            labels = np.argmax(log_resp, axis=1)
            centers = np.array(self.means_)  # samples x clusters
            centers_fill = centers[labels, :]

            assert centers_fill.shape == X.shape
            X[self.missing_d] = centers_fill[self.missing_d]

        super()._m_step(X, log_resp)  # Do the regular m step

        # Do sequence m step
        self.seq_dist.from_summaries(np.exp(log_resp))

    def fit(self, p_signal: pd.DataFrame):
        """
        Compute EM clustering.

        Args:
            p_signal: Dataframe of shape (number of peptides, number of samples)
                containing the phosphorylation signal. `p_signal.index` contains
                the length-11 AA sequence of each peptide, containing the
                phosphoacceptor in the middle and five AAs flanking it.
        """
        assert isinstance(p_signal, pd.DataFrame), "`p_signal` must be a pandas dataframe."
        sequences = p_signal.index.values
        assert (
            isinstance(sequences[0], str) and len(sequences[0]) == 11
        ), "The index of p_signal must be the peptide sequences of length 11"
        assert all(
            [token in AAlist for token in sequences[0]]
        ), "Sequence(s) contain invalid characters"
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

    def transform(self, as_df=False) -> np.ndarray | pd.DataFrame:
        """
        Return cluster centers.

        Args:
            as_df: Whether or not the result should be wrapped in a dataframe with labeled axes.

        Returns:
            The cluster centers, either a np array or pd df of shape (n_samples, n_components).
        """
        check_is_fitted(self, ["means_"])
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
        """
        p_signal = self.p_signal.copy()
        labels = self.labels()  # cluster assignments
        centers = self.transform()  # samples x clusters
        for ii in range(p_signal.shape[0]):
            p_signal[ii, np.isnan(p_signal[ii, :])] = centers[
                np.isnan(p_signal[ii, :]), labels[ii] - 1
            ]
        assert np.all(np.isfinite(p_signal))
        return p_signal

    def get_pssms(
        self, PsP_background=False, clusters: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
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
        PsP_background=True,
    ) -> np.ndarray:
        """Compute matrix-matrix similarity between kinase specificity profiles
        and cluster PSSMs to identify upstream kinases regulating clusters."""
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

    def has_empty_clusters(self) -> bool:
        """
        Checks whether the most recent call to fit() resulted in empty clusters.
        """
        check_is_fitted(self, ["scores_"])
        return np.unique(self.labels()).size != self.n_components

    def predict(self) -> np.ndarray[int]:
        """Provided the current model parameters, predict the cluster each peptide belongs to."""
        check_is_fitted(self, ["scores_"])
        return np.argmax(self.scores_, axis=1)

    def labels(self) -> np.ndarray[int]:
        """Find cluster assignment with highest likelihood for each peptide."""
        return self.predict()

    def score(self) -> float:
        """Generate score of the fitting."""
        check_is_fitted(self, ["lower_bound_"])
        return self.lower_bound_


def get_pspl_pssm_distances(
    pspls: np.ndarray,
    pssms: np.ndarray,
    as_df=False,
    pssm_names: Sequence[str] = None,
    kinases: Sequence[str] = None,
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
        dists = pd.DataFrame(dists, index=kinases, columns=pssm_names)
    return dists
