""" Clustering functions. """

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from .expectation_maximization import EM_clustering_opt, GenerateSeqBackgroundAndPAMscores
from .motifs import ForegroundSeqs
from .binomial import GenerateBPM
from .binomial import assignPeptidesBN
from .pam250 import assignPeptidesPAM


# pylint: disable=W0201


class MassSpecClustering(BaseEstimator):
    """ Cluster peptides by both sequence similarity and data behavior following an
    expectation-maximization algorithm. SeqWeight specifies which method's expectation step
    should have a larger effect on the peptide assignment. """

    def __init__(self, info, ncl, SeqWeight, distance_method, max_n_iter=1000, n_runs=1):
        self.info = info
        self.ncl = ncl
        self.SeqWeight = SeqWeight
        self.distance_method = distance_method
        self.max_n_iter = max_n_iter
        self.n_runs = n_runs

    def fit(self, X, _):
        """Compute EM clustering"""
        self.cl_seqs_, self.labels_, self.scores_, self.n_iter_, self.gmm_, self.wins_ = EM_clustering_opt(
            X, self.info, self.ncl, self.SeqWeight, self.distance_method, self.max_n_iter, self.n_runs
        )
        return self

    def transform(self, X):
        """Calculate cluster averages"""
        check_is_fitted(self, ["cl_seqs_", "labels_", "scores_", "n_iter_", "gmm_", "wins_"])

        centers, _ = ClusterAverages(X, self.labels_)
        return centers

    def clustermembers(self, X):
        """Generate dictionary containing peptide names and sequences for each cluster"""
        check_is_fitted(self, ["cl_seqs_", "labels_", "scores_", "n_iter_", "gmm_", "wins_"])

        _, clustermembers = ClusterAverages(X, self.labels_)
        return clustermembers

    def runSeqScore(self, sequences, cl_seqs):
        background = GenerateSeqBackgroundAndPAMscores(sequences, self.distance_method)

        if self.distance_method == "Binomial":
            binoM = GenerateBPM(cl_seqs, background)
            seq_scores = assignPeptidesBN(self.ncl, sequences, cl_seqs, background, binoM, self.labels_)
        else:
            seq_scores = assignPeptidesPAM(self.ncl, sequences, cl_seqs, background, self.labels_)

        return seq_scores

    def predict(self, data, sequences, _Y=None):
        """Provided the current model parameters, predict the cluster each peptide belongs to"""
        check_is_fitted(self, ["cl_seqs_", "labels_", "scores_", "n_iter_", "gmm_", "wins_"])
        seqs = ForegroundSeqs(sequences)
        cl_seqs = [ForegroundSeqs(self.cl_seqs_[i]) for i in range(self.ncl)]
        gmmp = self.gmm_.predict_proba(data.T)
        seq_scores = self.runSeqScore(seqs, cl_seqs)
        final_scores = seq_scores * self.SeqWeight + gmmp

        return np.argmax(final_scores, axis=1)

    def score(self, data, sequences, _Y=None):
        """Generate score of the fitting. If PAM250, the score is the averaged PAM250 score across peptides. If Binomial,
        the score is the mean binomial p-value across peptides"""
        check_is_fitted(self, ["cl_seqs_", "labels_", "scores_", "n_iter_", "gmm_", "wins_"])
        seqs = ForegroundSeqs(sequences)
        cl_seqs = [ForegroundSeqs(self.cl_seqs_[i]) for i in range(self.ncl)]
        gmmp = self.gmm_.predict_proba(data.T)
        seq_scores = self.runSeqScore(seqs, cl_seqs)
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

    members = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_clustermembers.items()]))
    return pd.DataFrame(centers).T, members
