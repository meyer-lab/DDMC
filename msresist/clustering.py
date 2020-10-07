""" Clustering functions. """

import glob
from copy import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from .expectation_maximization import EM_clustering_repeat
from .motifs import ForegroundSeqs
from .binomial import Binomial, position_weight_matrix, AAlist
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

        if distance_method == "PAM250":
            self.dist = PAM250(info, background, SeqWeight)
        elif distance_method == "Binomial":
            self.dist = Binomial(info, background, SeqWeight)

    def fit(self, X, y=None, nRepeats=3):
        """Compute EM clustering"""
        self.avgScores_, self.scores_, self.seq_scores_, self.gmm_ = EM_clustering_repeat(nRepeats, X, self.info, self.ncl, self.dist)

        # Use only to pickle model
        # if self.distance_method == "PAM250":
        #     for ii in range(self.ncl):
        #         self.gmm_.distributions[ii][-1].background = None
        #     self.dist.background = None

        return self

    def wins(self, X):
        """Find similarity of fitted model to data and sequence models"""
        check_is_fitted(self, ["scores_", "seq_scores_", "gmm_"])

        wDist = copy(self.dist)
        wDist.SeqWeight = 0.0
        data_model = EM_clustering_repeat(3, X, self.info, self.ncl, wDist)[1]
        wDist.SeqWeight = 10.0
        seq_model = EM_clustering_repeat(3, X, self.info, self.ncl, wDist)[1]

        dataDist = np.linalg.norm(self.scores_ - data_model)
        seqDist = np.linalg.norm(self.scores_ - seq_model)

        for _ in range(self.ncl - 1):
            data_model = np.roll(data_model, 1, axis=1)
            seq_model = np.roll(seq_model, 1, axis=1)

            dataDistTemp = np.linalg.norm(self.scores_ - data_model)
            seqDistTemp = np.linalg.norm(self.scores_ - seq_model)

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

    def pssms(self, bg_sequences):
        """Compute position-specific scoring matrix of each cluster."""
        pssms = []
        back_pssm = np.zeros((len(AAlist), 11), dtype=float)
        for ii in range(self.ncl):
            pssm = np.zeros((len(AAlist), 11), dtype=float)
            for jj, seq in enumerate(self.info["Sequence"]):
                seq = seq.upper()
                for kk, aa in enumerate(seq):
                    pssm[AAlist.index(aa), kk] += self.scores_[jj, ii]
                    if ii == 0:
                        back_pssm[AAlist.index(aa), kk] += 1.0

            # Normalize by position across residues and remove negative outliers
            for pos in range(pssm.shape[1]):
                pssm[:, pos] /= np.mean(pssm[:, pos])
                if ii == 0:
                    back_pssm[:, pos] /= np.mean(back_pssm[:, pos])
            pssm = np.log2(pssm)
            if ii == 0:
                back_pssm = np.log2(back_pssm)
            pssm -= back_pssm.copy()
            pssm = np.nan_to_num(pssm)
            pssm[pssm < -4] = -4
            pssm = pd.DataFrame(pssm)
            pssm.index = AAlist
            pssms.append(pssm)

        return pssms

    def predict_UpstreamKinases(self, bg_sequences):
        """Compute matrix-matrix similarity between kinase specificity profiles and cluster PSSMs to identify upstream kinases regulating clusters."""
        PSPLs = PSPSLdict()
        bg_prob, PSSMs = TransformKinasePredictionMats(self.pssms(bg_sequences), bg_sequences)
        a = np.zeros((len(PSPLs), len(PSSMs)))

        for ii, spec_profile in enumerate(PSPLs.values()):
            sp = np.log10(np.power(2, spec_profile) / bg_prob)
            sp -= np.mean(sp)
            for jj, pssm in enumerate(PSSMs):
                pssm -= np.mean(pssm)
                a[ii, jj] = np.linalg.norm(pssm - sp)

        table = pd.DataFrame(a)
        table.insert(0, "Kinase", list(PSPSLdict().keys()))
        return table

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


def ClusterAverages(X, labels):
    """Calculate cluster averages and dictionary with cluster members and sequences"""
    X = X.T.assign(cluster=labels)
    centers = []
    dict_clustermembers = {}
    for i in range(0, max(labels) + 1):
        centers.append(list(X[X["cluster"] == i].iloc[:, :-1].mean()))
        if "object" in list(X.dtypes):
            dict_clustermembers["Protein_C" + str(i + 1)] = list(X[X["cluster"] == i]["Protein"])
            dict_clustermembers["Gene_C" + str(i + 1)] = list(X[X["cluster"] == i]["Gene"])
            dict_clustermembers["Sequence_C" + str(i + 1)] = list(X[X["cluster"] == i]["Sequence"])
            dict_clustermembers["Position_C" + str(i + 1)] = list(X[X["cluster"] == i]["Position"])

    members = pd.DataFrame({k: pd.Series(v) for (k, v) in dict_clustermembers.items()})
    return pd.DataFrame(centers).T, members


def PSPSLdict():
    """Generate dictionary with kinase name-specificity profile pairs"""
    pspl_dict = {}
    PSPLs = glob.glob("./msresist/data/PSPL/*.csv")
    for sp in PSPLs:
        sp_mat = pd.read_csv(sp)

        if sp_mat.shape[0] > 20:  # Remove profiling of fixed pY and pT, include only natural AA
            sp_mat = sp_mat.iloc[:-2, 1:].values
        else:
            sp_mat = sp_mat.iloc[:, 1:].values

        pspl_dict[sp.split("PSPL/")[1].split(".csv")[0]] = sp_mat
    return pspl_dict


def TransformKinasePredictionMats(PSSMs, bg_sequences):
    """Transform PSSMs and PSPLs to perform matrix math."""
    bg_prob = np.array(list(position_weight_matrix(ForegroundSeqs(bg_sequences)).values()))
    bg_prob = np.delete(bg_prob, [5, 10], 1)  # Remove P0 and P+5 from background
    PSSMs = [np.delete(np.array(list(mat)), [5, 10], 1) for mat in PSSMs]  # Remove P0 and P+5 from pssms
    return bg_prob, PSSMs
