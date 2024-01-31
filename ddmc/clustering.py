""" Clustering functions. """

from typing import Literal, List, Dict
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
    expectation-maximization algorithm. SeqWeight specifies which method's expectation step
    should have a larger effect on the peptide assignment."""

    def __init__(
        self,
        sequences,
        n_components: int,
        seq_weight: float,
        distance_method: Literal["PAM250", "Binomial"],
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
        self.gen_peptide_distances(sequences, distance_method)
        self.seq_weight = seq_weight

    def gen_peptide_distances(self, seqs: np.ndarray | pd.DataFrame, distance_method):
        # store parameters for sklearn's checks
        self.distance_method = distance_method
        if not isinstance(seqs, np.ndarray):
            seqs = seqs.values
        if seqs.dtype != str:
            seqs = seqs.astype("str")
        seqs = np.char.upper(seqs)
        self.sequences = seqs
        if distance_method == "PAM250":
            self.seq_dist: PAM250 | Binomial = PAM250(seqs)
        elif distance_method == "Binomial":
            self.seq_dist = Binomial(seqs)
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

    def fit(self, X: pd.DataFrame):
        """
        Compute EM clustering

        Args:
            X: dataframe consisting of a "Sequence" column, and sample
                columns. Every column that is not named "Sequence" will be treated
                as a sample.
        """
        # TODO: assert that the peptides passed in match the length of X
        # TODO: probably just pass in sequences here
        d = np.array(X)

        if np.any(np.isnan(d)):
            self._missing = True
            self.missing_d = np.isnan(d)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = SoftImpute(verbose=False).fit_transform(d)
        else:
            self._missing = False

        super().fit(d)
        self.scores_ = self.predict_proba(d)

        assert np.all(np.isfinite(self.scores_))
        assert np.all(np.isfinite(self.seq_scores_))
        return self

    def wins(self, X):
        """Find similarity of fitted model to data and sequence models"""
        check_is_fitted(self, ["scores_", "seq_scores_"])

        alt_model = deepcopy(self)
        alt_model.seq_weight = 0.0  # No influence
        alt_model.fit(X)
        data_model = alt_model.scores_

        alt_model.seq_weight = 1000.0  # Overwhelming influence
        alt_model.fit(X)
        seq_model = alt_model.scores_

        dataDist = np.linalg.norm(self.scores_ - data_model)
        seqDist = np.linalg.norm(self.scores_ - seq_model)

        for i in itertools.permutations(np.arange(self.n_components)):
            dataDistTemp = np.linalg.norm(self.scores_ - data_model[:, i])
            seqDistTemp = np.linalg.norm(self.scores_ - seq_model[:, i])

            dataDist = np.minimum(dataDist, dataDistTemp)
            seqDist = np.minimum(seqDist, seqDistTemp)

        return (dataDist, seqDist)

    def transform(self) -> np.ndarray:
        """Calculate cluster averages."""
        check_is_fitted(self, ["means_"])
        return self.means_.T

    def impute(self, X: np.ndarray) -> np.ndarray:
        """Impute a matching dataset."""
        X = X.copy()
        labels = self.labels()  # cluster assignments
        centers = self.transform()  # samples x clusters

        assert len(labels) == X.shape[0]
        for ii in range(X.shape[0]):  # X is peptides x samples
            X[ii, np.isnan(X[ii, :])] = centers[np.isnan(X[ii, :]), labels[ii] - 1]

        assert np.all(np.isfinite(X))
        return X

    def get_pssms(self, PsP_background=False, clusters: List=None):
        """Compute position-specific scoring matrix of each cluster.
        Note, to normalize by amino acid frequency this uses either
        all the sequences in the data set or a collection of random MS phosphosites in PhosphoSitePlus.
        """
        pssms, pssm_names = [], []
        if PsP_background:
            bg_seqs = BackgroundSeqs(self.sequences)
            back_pssm = compute_control_pssm(bg_seqs)
        else:
            back_pssm = np.zeros((len(AAlist), 11), dtype=float)
        for ii in range(1, self.n_components + 1):
            # Check for empty clusters and ignore them, if there are
            l1 = list(np.arange(self.n_components) + 1)
            l2 = list(set(self.labels()))
            ec = [i for i in l1 + l2 if i not in l1 or i not in l2]
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
            return pssms[[np.where(pssm_names == cluster)[0][0] for cluster in clusters]]

        return pssm_names, pssms

    def predict_upstream_kinases(
        self,
        PsP_background=True,
    ):
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

    def predict(self) -> np.ndarray:
        """Provided the current model parameters, predict the cluster each peptide belongs to."""
        check_is_fitted(self, ["scores_"])
        return np.argmax(self.scores_, axis=1)

    def labels(self) -> np.ndarray:
        """Find cluster assignment with highest likelihood for each peptide."""
        return self.predict() + 1

    def score(self) -> float:
        """Generate score of the fitting."""
        check_is_fitted(self, ["lower_bound_"])
        return self.lower_bound_


def get_pspl_pssm_distances(
    pspls: np.ndarray, pssms: np.ndarray, as_df=False, pssm_names=None, kinases=None
) -> np.ndarray | pd.DataFrame:
    """
    Args:
        pspls: kinase specificity profiles of shape (n_kinase, 20, 9)
        pssms: position-specific scoring matrices of shape (n_peptides, 20, 11) 
    """
    assert pssms.shape[1:3] == (20, 11)
    assert pspls.shape[1:3] == (20, 9)
    pssms = np.delete(pssms, [5, 10], axis=2)
    dists = np.linalg.norm(pspls[:, None, :, :] - pssms[None, :, :, :], axis=(2, 3))
    if as_df:
        dists = pd.DataFrame(dists, index=kinases, columns=pssm_names)
    return dists
