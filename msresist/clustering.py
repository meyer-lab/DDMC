""" Clustering functions. """

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from Bio import motifs
from .expectation_maximization import EM_clustering
from .motifs import ForegroundSeqs
from .binomial import assignPeptidesBN, position_weight_matrix, AAlist, BackgroundSeqs, GenerateBinarySeqID
from .pam250 import assignPeptidesPAM, MotifPam250Scores


# pylint: disable=W0201


class MassSpecClustering(BaseEstimator):
    """ Cluster peptides by both sequence similarity and data behavior following an
    expectation-maximization algorithm. SeqWeight specifies which method's expectation step
    should have a larger effect on the peptide assignment. """

    def __init__(self, info, ncl, SeqWeight, distance_method, max_n_iter=500, background=False, bg_mat=False, dataTensor=False):
        self.info = info
        self.ncl = ncl
        self.SeqWeight = SeqWeight
        self.distance_method = distance_method
        self.max_n_iter = max_n_iter
        self.background = background
        self.bg_mat = bg_mat
        self.dataTensor = dataTensor

    def fit(self, X, y=None):
        """Compute EM clustering"""
        self.scores_, self.seq_scores_, self.gmm_ = EM_clustering(
            X, self.info, self.ncl, self.SeqWeight, self.distance_method, self.background, self.bg_mat, self.dataTensor, self.max_n_iter
        )
        return self

    def wins(self, d):
        """Find the sequence, data, both, and mix wins of the fitted model"""
        check_is_fitted(self, ["scores_", "seq_scores_", "gmm_"])

        labels_ = self.labels()
        SeqIdx = np.argmax(self.seq_scores_, axis=1)
        DataIdx = self.gmm_.predict(d.T)

        SeqWins = np.sum((SeqIdx == labels_) & (DataIdx != labels_))
        DataWins = np.sum((DataIdx == labels_) & (SeqIdx != labels_))
        BothWin = np.sum((DataIdx == labels_) & (SeqIdx == labels_))
        MixWins = np.sum((DataIdx != labels_) & (SeqIdx != labels_))

        return (SeqWins, DataWins, BothWin, MixWins)

    def labels(self):
        """Find cluster assignments"""
        check_is_fitted(self, ["scores_", "seq_scores_", "gmm_"])

        return np.argmax(self.scores_, axis=1)

    def cl_seqs(self, sequences):
        """Return cluster sequences"""
        check_is_fitted(self, ["scores_", "seq_scores_", "gmm_"])

        labels_ = self.labels()
        return [list(sequences.iloc[np.squeeze(np.argwhere(labels_ == i))]) for i in range(self.ncl)]

    def transform(self, X):
        """Calculate cluster averages"""
        check_is_fitted(self, ["scores_", "seq_scores_", "gmm_"])

        centers, _ = ClusterAverages(X, self.labels())
        return centers

    def clustermembers(self, X):
        """Generate dictionary containing peptide names and sequences for each cluster"""
        check_is_fitted(self, ["scores_", "seq_scores_", "gmm_"])

        _, clustermembers = ClusterAverages(X, self.labels())
        return clustermembers

    def pssms(self, bg_sequences):
        """Compute position-specific scoring matrix of each cluster."""
        bg_seqs = ForegroundSeqs(bg_sequences)
        bg_freqs = motifs.create(bg_seqs).counts
        cl_seqs_ = self.cl_seqs(bg_sequences)

        AAfreq_IS = {}
        for i in range(20):
            AAfreq_IS[list(bg_freqs.keys())[i]] = np.sum(bg_freqs[i]) / (len(bg_seqs) * len(bg_seqs[0]))

        pssms = []
        for j in range(self.ncl):
            pssms.append(motifs.create(ForegroundSeqs(cl_seqs_[j])).counts.normalize(pseudocounts=AAfreq_IS).log_odds(AAfreq_IS))

        return pssms

    def runSeqScore(self, sequences):
        if self.distance_method == "Binomial":
            background = position_weight_matrix(BackgroundSeqs(sequences))
            bg_mat = np.array([background[AA] for AA in AAlist])
            dataTensor = GenerateBinarySeqID(sequences)
            seq_scores = assignPeptidesBN(dataTensor, self.scores_, bg_mat)
        else:
            background = MotifPam250Scores(sequences)
            seq_scores = assignPeptidesPAM(self.ncl, self.scores_, background)

        return seq_scores

    def predict(self, data, sequences, _Y=None):
        """Provided the current model parameters, predict the cluster each peptide belongs to"""
        check_is_fitted(self, ["scores_", "seq_scores_", "gmm_"])
        gmmp = self.gmm_.predict_proba(data.T)
        seq_scores = self.runSeqScore(sequences)
        final_scores = seq_scores * self.SeqWeight + gmmp

        return np.argmax(final_scores, axis=1)

    def score(self, data, sequences, _Y=None):
        """Generate score of the fitting. If PAM250, the score is the averaged PAM250 score across peptides. If Binomial,
        the score is the mean binomial p-value across peptides"""
        check_is_fitted(self, ["scores_", "seq_scores_", "gmm_"])
        gmmp = self.gmm_.predict_proba(data.T)
        seq_scores = self.runSeqScore(sequences)
        final_scores = seq_scores * self.SeqWeight + gmmp

        return np.mean(np.max(final_scores, axis=1))

    def get_params(self, deep=True):
        """Returns a dict of the estimator parameters with their values"""
        return {
            "info": self.info,
            "ncl": self.ncl,
            "SeqWeight": self.SeqWeight,
            "distance_method": self.distance_method,
            "max_n_iter": self.max_n_iter,
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
    #             dict_clustermembers["UniprotAcc_C" + str(i + 1)] = list(X[X["cluster"] == i]["UniprotAcc"])
    #             dict_clustermembers["r2/Std_C" + str(i + 1)] = list(X[X["cluster"] == i]["r2_Std"])
    #             dict_clustermembers["BioReps_C" + str(i + 1)] = list(X[X["cluster"] == i]["BioReps"])

    members = pd.DataFrame({k: pd.Series(v) for (k, v) in dict_clustermembers.items()})
    return pd.DataFrame(centers).T, members
