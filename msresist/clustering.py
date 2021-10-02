""" Clustering functions. """

import warnings
import glob
from copy import deepcopy
import itertools
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted
from .binomial import Binomial, AAlist, BackgroundSeqs, frequencies
from .pam250 import PAM250
from fancyimpute import SoftImpute


# pylint: disable=W0201


class MassSpecClustering(GaussianMixture):
    """ Cluster peptides by both sequence similarity and data behavior following an
    expectation-maximization algorithm. SeqWeight specifies which method's expectation step
    should have a larger effect on the peptide assignment. """
    def __init__(self, info, ncl, SeqWeight, distance_method, random_state=None):
        super().__init__(n_components=ncl, covariance_type="diag", n_init=2, max_iter=200, tol=1e-4, random_state=random_state)

        self.info = info
        self.SeqWeight = SeqWeight
        self.distance_method = distance_method

        seqs = [s.upper() for s in info["Sequence"]]

        if distance_method == "PAM250":
            self.seqDist = PAM250(seqs)
        elif distance_method == "Binomial":
            self.seqDist = Binomial(info["Sequence"], seqs)
        else:
            raise ValueError("Wrong distance type.")

    def _estimate_log_prob(self, X):
        """ Estimate the log-probability of each point in each cluster. """
        logp = super()._estimate_log_prob(X) # Do the regular work

        # Add in the sequence effect
        self.seq_scores_ = self.SeqWeight * self.seqDist.logWeights
        logp += self.seq_scores_

        return logp

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        super()._m_step(X, log_resp) # Do the regular m step

        # Do sequence m step
        self.seqDist.from_summaries(np.exp(log_resp))

    def fit(self, X, y=None):
        """Compute EM clustering"""
        d = np.array(X.T)

        if np.any(np.isnan(d)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = SoftImpute(verbose=False).fit_transform(d)

            assert np.all(np.isfinite(d))
            imputt = True
        else:
            imputt = False

        super().fit(d)
        self.scores_ = self.predict_proba(d)

        if imputt:
            d = np.array(X.T)
            d = self.impute(d)
            super().fit(d)
            self.scores_ = self.predict_proba(d)

        assert np.all(np.isfinite(self.scores_))
        assert np.all(np.isfinite(self.seq_scores_))
        return self

    def wins(self, X):
        """Find similarity of fitted model to data and sequence models"""
        check_is_fitted(self, ["scores_", "seq_scores_"])

        alt_model = deepcopy(self)
        alt_model.SeqWeight = 0.0 # No influence
        alt_model.fit(X)
        data_model = alt_model.scores_

        alt_model.SeqWeight = 50.0 # Overwhelming influence
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

    def transform(self):
        """ Calculate cluster averages. """
        check_is_fitted(self, ["means_"])
        return self.means_.T

    def impute(self, X):
        """ Impute a matching dataset. """
        X = X.copy()
        labels = self.labels() # cluster assignments
        centers = self.transform() # samples x clusters

        assert len(labels) == X.shape[0]
        for ii in range(X.shape[0]): # X is peptides x samples
            X[ii, np.isnan(X[ii, :])] = centers[np.isnan(X[ii, :]), labels[ii] - 1]

        assert np.all(np.isfinite(X))
        return X

    def pssms(self, PsP_background=False):
        """Compute position-specific scoring matrix of each cluster.
        Note, to normalize by amino acid frequency this uses either
        all the sequences in the data set or a collection of random MS phosphosites in PhosphoSitePlus."""
        pssms, cl_num = [], []
        if PsP_background:
            bg_seqs = BackgroundSeqs(self.info["Sequence"])
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
            for jj, seq in enumerate(self.info["Sequence"]):
                seq = seq.upper()
                for kk, aa in enumerate(seq):
                    pssm[AAlist.index(aa), kk] += self.scores_[jj, ii - 1]
                    if ii == 1 and not PsP_background:
                        back_pssm[AAlist.index(aa), kk] += 1.0

            # Normalize by position across residues and remove negative outliers
            for pos in range(pssm.shape[1]):
                if pos == 5:
                    continue
                pssm[:, pos] /= np.mean(pssm[:, pos])
                if ii == 1 and not PsP_background:
                    back_pssm[:, pos] /= np.mean(back_pssm[:, pos])
            pssm = np.ma.log2(pssm)
            pssm = pssm.filled(0)
            if ii == 1 and not PsP_background:
                back_pssm = np.ma.log2(back_pssm)
                back_pssm = back_pssm.filled(0)
            pssm -= back_pssm.copy()
            pssm = np.nan_to_num(pssm)
            pssm = pd.DataFrame(pssm)
            pssm.index = AAlist

            # Normalize phosphoacceptor position to frequency
            df = pd.DataFrame(self.info["Sequence"].str.upper())
            df["Cluster"] = self.labels()
            clSeq = df[df["Cluster"] == ii]["Sequence"]
            clSeq = pd.DataFrame(frequencies(clSeq)).T
            tm = np.mean([clSeq.loc["S", 5], clSeq.loc["T", 5], clSeq.loc["Y", 5]])
            for p_site in ["S", "T", "Y"]:
                pssm.loc[p_site, 5] = np.log2(clSeq.loc[p_site, 5] / tm)

            pssms.append(np.clip(pssm, a_min=0, a_max=3))
            cl_num.append(ii)

        return pssms, cl_num

    def predict_UpstreamKinases(self, additional_pssms=False):
        """Compute matrix-matrix similarity between kinase specificity profiles and cluster PSSMs to identify upstream kinases regulating clusters."""
        PSPLs = PSPLdict()
        PSSMs, cl_num = self.pssms(PsP_background=True)

        # Optionally add external pssms
        if not isinstance(additional_pssms, bool):
            PSSMs += additional_pssms
        PSSMs = [np.delete(np.array(list(np.array(mat))), [5, 10], axis=1) for mat in PSSMs]  # Remove P0 and P+5 from pssms

        a = np.zeros((len(PSPLs), len(PSSMs)))
        for ii, spec_profile in enumerate(PSPLs.values()):
            for jj, pssm in enumerate(PSSMs):
                a[ii, jj] = np.linalg.norm(pssm - spec_profile)

        table = pd.DataFrame(a)
        table.columns = cl_num
        table.insert(0, "Kinase", list(PSPLdict().keys()))
        return table

    def predict(self):
        """ Provided the current model parameters, predict the cluster each peptide belongs to. """
        check_is_fitted(self, ["scores_"])
        return np.argmax(self.scores_, axis=1)

    def labels(self):
        """ Find cluster assignment with highest likelihood for each peptide. """
        return self.predict() + 1

    def score(self):
        """ Generate score of the fitting. """
        check_is_fitted(self, ["lower_bound_"])
        return self.lower_bound_

    def get_params(self, deep=True):
        """ Returns a dict of the estimator parameters with their values. """
        dictt = super().get_params(deep=deep)
        dictt["info"] = self.info
        dictt["SeqWeight"] = self.SeqWeight
        dictt["distance_method"] = self.distance_method
        return dictt


def PSPLdict():
    """Generate dictionary with kinase name-specificity profile pairs"""
    pspl_dict = {}
    # individual files
    PSPLs = glob.glob("./msresist/data/PSPL/*.csv")
    for sp in PSPLs:
        if sp == "./msresist/data/PSPL/pssm_data.csv":
            continue
        sp_mat = pd.read_csv(sp).sort_values(by="Unnamed: 0")

        if sp_mat.shape[0] > 20:  # Remove profiling of fixed pY and pT, include only natural AA
            assert np.all(sp_mat.iloc[:-2, 0] == AAlist), "aa don't match"
            sp_mat = sp_mat.iloc[:-2, 1:].values
        else:
            assert np.all(sp_mat.iloc[:, 0] == AAlist), "aa don't match"
            sp_mat = sp_mat.iloc[:, 1:].values

        if np.all(sp_mat >= 0):
            sp_mat = np.log2(sp_mat)

        pspl_dict[sp.split("PSPL/")[1].split(".csv")[0]] = sp_mat

    # NetPhores PSPL results
    f = pd.read_csv("msresist/data/PSPL/pssm_data.csv", header=None)
    matIDX = [np.arange(16) + i for i in range(0, f.shape[0], 16)]
    for ii in matIDX:
        kin = f.iloc[ii[0], 0]
        mat = f.iloc[ii[1:], :].T
        mat.columns = np.arange(mat.shape[1])
        mat = mat.iloc[:-1, 2:12].drop(8, axis=1).astype("float64").values
        mat = np.ma.log2(mat)
        mat = mat.filled(0)
        mat = np.clip(mat, a_min=0, a_max=3)
        pspl_dict[kin] = mat

    return pspl_dict


def compute_control_pssm(bg_sequences):
    """Generate PSSM."""
    back_pssm = np.zeros((len(AAlist), 11), dtype=float)
    for _, seq in enumerate(bg_sequences):
        for kk, aa in enumerate(seq):
            back_pssm[AAlist.index(aa), kk] += 1.0
    for pos in range(back_pssm.shape[1]):
        back_pssm[:, pos] /= np.mean(back_pssm[:, pos])
    back_pssm = np.ma.log2(back_pssm)
    return back_pssm.filled(0)


KinToPhosphotypeDict = {
    "ABL": "Y",
    "AKT": "S/T",
    "ALK": "Y",
    "BLK": "Y",
    "BRK": "Y",
    "CK2": "S/T",
    "ERK2": "S/T",
    "FRK": "Y",
    "HCK": "Y",
    "INSR": "Y",
    "LCK": "Y",
    "LYN": "Y",
    "MET": "Y",
    "NEK1": "S/T",
    "NEK2": "S/T",
    "NEK3": "S/T",
    "NEK4": "S/T",
    "NEK5": "S/T",
    "NEK6": "S/T",
    "NEK7": "S/T",
    "NEK8": "S/T",
    "NEK9": "S/T",
    "NEK10_S": "S/T",
    "NEK10_Y": "Y",
    "PKA": "S/T",
    "PKC-theta": "S/T",
    "PKD": "S/T",
    "PLM2": "S/T",
    "RET": "Y",
    "SRC": "Y",
    "TbetaRII": "S/T",
    "YES": "Y",
    "BRCA1": "S/T",
    "AMPK": "S/T",
    "CDK5": "S/T",
    "CK1": "S/T",
    "DMPK1": "S/T",
    "EGFR": "Y",
    "InsR": "Y",
    "p38": "S/T",
    "ERK1": "S/T",
    "SHC1": "Y",
    "SH2_PLCG1": "Y",
    "SH2_INPP5D": "Y",
    "SH2_SH3BP2": "Y",
    "SH2_SHC2": "Y",
    "SH2_SHE": "Y",
    "SH2_Syk": "Y",
    "SH2_TNS4": "Y",
    "CLK2": "S/T",
    "DAPK3": "S/T",
    "ICK": "S/T",
    "STK11": "S/T",
    "MST1": "S/T",
    "MST4": "S/T",
    "PAK2": "S/T",
    "Pim1": "S/T",
    "Pim2": "S/T",
    "SLK": "S/T",
    "TGFbR2": "S/T",
    "TLK1": "S/T",
    "TNIK": "S/T",
    "p70S6K": "S/T",
    "EphA3": "Y"
}
