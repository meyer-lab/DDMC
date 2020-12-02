""" Clustering functions. """

import glob
import itertools
from copy import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from .expectation_maximization import EM_clustering_repeat
from .motifs import ForegroundSeqs
from .binomial import Binomial, AAlist, BackgroundSeqs, frequencies
from .pam250 import PAM250


# pylint: disable=W0201


class MassSpecClustering(BaseEstimator):
    """ Cluster peptides by both sequence similarity and data behavior following an
    expectation-maximization algorithm. SeqWeight specifies which method's expectation step
    should have a larger effect on the peptide assignment. """

    def __init__(self, info, ncl, SeqWeight, distance_method, background=False):
        self.info = info
        self.ncl = ncl
        self.SeqWeight = SeqWeight
        self.distance_method = distance_method

        seqs = [s.upper() for s in info["Sequence"]]

        if distance_method == "PAM250":
            self.dist = PAM250(seqs, SeqWeight)
        elif distance_method == "Binomial":
            self.dist = Binomial(info["Sequence"], seqs, SeqWeight)

    def fit(self, X, y=None, nRepeats=3):
        """Compute EM clustering"""
        self.avgScores_, self.scores_, self.seq_scores_, self.gmm_ = EM_clustering_repeat(nRepeats, X, self.info, self.ncl, self.dist)

        return self

    def wins(self, X):
        """Find similarity of fitted model to data and sequence models"""
        check_is_fitted(self, ["scores_", "seq_scores_", "gmm_"])

        wDist = self.dist.copy()
        wDist.SeqWeight = 0.0
        data_model = EM_clustering_repeat(3, X, self.info, self.ncl, wDist)[1]
        wDist.SeqWeight = 10.0
        seq_model = EM_clustering_repeat(3, X, self.info, self.ncl, wDist)[1]

        dataDist = np.linalg.norm(self.scores_ - data_model)
        seqDist = np.linalg.norm(self.scores_ - seq_model)

        for i in itertools.permutations(np.arange(self.ncl)):
            dataDistTemp = np.linalg.norm(self.scores_ - data_model[:, i])
            seqDistTemp = np.linalg.norm(self.scores_ - seq_model[:, i])

            dataDist = np.minimum(dataDist, dataDistTemp)
            seqDist = np.minimum(seqDist, seqDistTemp)

        return (dataDist, seqDist)

    def transform(self):
        """Calculate cluster averages"""
        check_is_fitted(self, ["gmm_"])

        centers = np.zeros((self.ncl, self.gmm_.distributions[0].d - 1))

        for ii, distClust in enumerate(self.gmm_.distributions):
            for jj, dist in enumerate(distClust[:-1]):
                centers[ii, jj] = dist.parameters[0]

        return centers.T

    def labels(self):
        """Find cluster assignment with highest likelihood for each peptide"""
        check_is_fitted(self, ["gmm_"])

        return np.argmax(self.scores_, axis=1) + 1

    def pssms(self, PsP_background=False):
        """Compute position-specific scoring matrix of each cluster.
        Note, to normalize by amino acid frequency this uses either
        all the sequences in the data set or a collection of random MS phosphosites in PhosphoSitePlus."""
        pssms = []
        if PsP_background:
            bg_seqs = BackgroundSeqs(self.info["Sequence"])
            back_pssm = background_pssm(bg_seqs)
        else:
            back_pssm = np.zeros((len(AAlist), 11), dtype=float)
        for ii in range(self.ncl):
            pssm = np.zeros((len(AAlist), 11), dtype=float)
            for jj, seq in enumerate(self.info["Sequence"]):
                seq = seq.upper()
                for kk, aa in enumerate(seq):
                    pssm[AAlist.index(aa), kk] += self.scores_[jj, ii]
                    if ii == 0 and not PsP_background:
                        back_pssm[AAlist.index(aa), kk] += 1.0

            # Normalize by position across residues and remove negative outliers
            for pos in range(pssm.shape[1]):
                if pos == 5:
                    continue
                pssm[:, pos] /= np.mean(pssm[:, pos])
                if ii == 0 and not PsP_background:
                    back_pssm[:, pos] /= np.mean(back_pssm[:, pos])
            pssm = np.log2(pssm)
            if ii == 0 and not PsP_background:
                back_pssm = np.log2(back_pssm)
            pssm -= back_pssm.copy()
            pssm = np.nan_to_num(pssm)
            pssm = pd.DataFrame(pssm)
            pssm.index = AAlist

            # Normalize phosphoacceptor position to frequency
            df = pd.DataFrame(self.info["Sequence"].str.upper())
            df["Cluster"] = self.labels()
            clSeq = df[df["Cluster"] == ii + 1]["Sequence"]
            clSeq = pd.DataFrame(frequencies(clSeq)).T
            tm = np.mean([clSeq.loc["S", 5], clSeq.loc["T", 5], clSeq.loc["Y", 5]])
            for p_site in ["S", "T", "Y"]:
                pssm.loc[p_site, 5] = np.log2(clSeq.loc[p_site, 5] / tm)

            pssm[pssm < -3] = -3
            pssms.append(pssm)

        return pssms

    def predict_UpstreamKinases(self, n_components=2):
        # Split PSPLs into pY and pST
        pspls = PSPSLdict()
        kins = list(PSPSLdict().keys())

        sp_y = {}
        sp_st = {}
        for kin in kins:
            if KinToPhosphotypeDict[kin] == "Y":
                sp_y[kin] = pspls[kin]
            else:
                sp_st[kin] = pspls[kin]

        # Split PSSMs into pY and pST
        pssm_y = {}
        pssm_st = {}
        for ii, mat in enumerate(self.pssms(PsP_background=True)):
            mat_ = np.delete(list(np.array(mat)), [5, 10], 1)
            if mat.iloc[:, 5].idxmax() == "Y":
                pssm_y[ii + 1] = mat_
            else:
                pssm_st[ii + 1] = mat_

        mats_y = list(sp_y.values()) + list(pssm_y.values())
        mats_st = list(sp_st.values()) + list(pssm_st.values())
        mats = [mats_y, mats_st]
        sp = [len(sp_y), len(sp_st)]
        mot = [len(pssm_y), len(pssm_st)]
        labels = [list(sp_y.keys()) + list(pssm_y.keys()), list(sp_st.keys()) + list(pssm_st.keys())]
        pX = ["Y", "S/T"]

        tables = []
        for q, mat in enumerate(mats):
            n = len(mat)
            res = np.empty((n, n), dtype=float)
            for ii in range(n):
                for jj in range(n):
                    res[ii, jj] = np.linalg.norm(mat[ii] - mat[jj])
            res[res < 1.0e-100] = 0
            seed = np.random.RandomState(seed=3)
            mds = MDS(n_components=n_components, max_iter=3000, eps=1e-9, random_state=seed,
                      dissimilarity="precomputed", n_jobs=1)
            pos = mds.fit(res).embedding_

            nmds = MDS(n_components=n_components, metric=False, max_iter=3000, eps=1e-12,
                       dissimilarity="precomputed", random_state=seed, n_jobs=1,
                       n_init=1)
            npos = nmds.fit_transform(res, init=pos)

            clf = PCA(n_components=n_components)
            npos = clf.fit_transform(npos)

            t = pd.DataFrame()
            for i in range(n_components):
                c = str(i + 1)
                t["PC" + c] = npos[:, i]
            t["Matrix type"] = ["PSPL"] * sp[q] + ["PSSM"] * mot[q]
            t["Label"] = labels[q]
            t["pX"] = pX[q]
            tables.append(t)

        return tables

    def predict(self):
        """Provided the current model parameters, predict the cluster each peptide belongs to"""
        check_is_fitted(self, ["scores_"])
        return np.argmax(self.scores_, axis=1)

    def score(self):
        """ Generate score of the fitting. """
        check_is_fitted(self, ["avgScores_"])
        return self.avgScores_

    def get_params(self, deep=True):
        """Returns a dict of the estimator parameters with their values"""
        return {
            "info": self.info,
            "ncl": self.ncl,
            "SeqWeight": self.SeqWeight,
            "distance_method": self.distance_method
        }

    def set_params(self, **parameters):
        """Necessary to make this estimator scikit learn-compatible"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def PSPSLdict():
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
        mat = np.log2(mat)
        mat[mat > 3] = 3
        mat[mat < -3] = -3
        pspl_dict[kin] = mat

    return pspl_dict


def background_pssm(bg_sequences):
    """Generate PSSM of PhosphoSitePlus phosphosite sequences."""
    back_pssm = np.zeros((len(AAlist), 11), dtype=float)
    for _, seq in enumerate(bg_sequences):
        for kk, aa in enumerate(seq):
            back_pssm[AAlist.index(aa), kk] += 1.0
    for pos in range(back_pssm.shape[1]):
        back_pssm[:, pos] /= np.mean(back_pssm[:, pos])
    return np.log2(back_pssm)


KinToPhosphotypeDict = {
    "ABL": "Y",
    "AKT": "S/T",
    "ALK": "Y",
    "BLK": "Y",
    "BRK": "Y",
    "CKII": "S/T",
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
    "LKB1": "S/T",
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
