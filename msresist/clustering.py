""" Clustering functions. """

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from msresist.sequence_analysis import EM_clustering_opt, e_step


class MassSpecClustering(BaseEstimator):
    """ Cluster peptides by both sequence similarity and data behavior following an
    expectation-maximization algorithm. SeqWeight specifies which method's expectation step
    should have a larger effect on the peptide assignment. """

    def __init__(self, info, ncl, SeqWeight, distance_method, max_n_iter=100000, n_runs=5):
        self.info = info
        self.ncl = ncl
        self.SeqWeight = SeqWeight
        self.distance_method = distance_method
        self.max_n_iter = max_n_iter
        self.n_runs = n_runs

    def fit(self, X, _):
        """ Compute EM clustering. """
        self.cl_seqs_, self.labels_, self.scores_, self.n_iter_, self.gmmp = EM_clustering_opt(X, self.info,
                                                                                               self.ncl,
                                                                                               self.SeqWeight,
                                                                                               self.distance_method,
                                                                                               self.max_n_iter,
                                                                                               self.n_runs)
        return self

    def transform(self, X):
        """ calculate cluster averages. """
        check_is_fitted(self, ["cl_seqs_", "gmmp", "labels_", "scores_", "n_iter_"])

        centers, _ = ClusterAverages(X, self.labels_)
        return centers

    def clustermembers(self, X):
        """ generate dictionary containing peptide names and sequences for each cluster. """
        check_is_fitted(self, ["cl_seqs_", "gmmp", "labels_", "scores_", "n_iter_"])

        _, clustermembers = ClusterAverages(X, self.labels_)
        return clustermembers

    def predict(self, X, _Y=None):
        """ Predict the cluster each sequence in ABC belongs to. If this estimator is gridsearched alone it
        won't work since all sequences are passed. """
        check_is_fitted(self, ["cl_seqs_", "gmmp", "labels_", "scores_", "n_iter_"])

        labels, _ = e_step(X, self.cl_seqs_, self.gmmp, self.distance_method, self.SeqWeight, self.ncl)
        return labels

    def score(self, X, _Y=None):
        """ Scoring method, mean of combined p-value of all peptides"""
        check_is_fitted(self, ["cl_seqs_", "gmmp", "labels_", "scores_", "n_iter_"])

        _, scores = e_step(X, self.cl_seqs_, self.gmmp, self.distance_method, self.SeqWeight, self.ncl)
        return scores

    def get_params(self, deep=True):
        """ Returns a dict of the estimator parameters with their values. """
        return {"info": self.info, "ncl": self.ncl,
                "SeqWeight": self.SeqWeight, "distance_method": self.distance_method,
                "max_n_iter": self.max_n_iter}

    def set_params(self, **parameters):
        """ Necessary to make this estimator scikit learn-compatible."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def ClusterAverages(X, labels):
    """ calculate cluster averages and dictionary with cluster members and sequences. """
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

    return pd.DataFrame(centers).T, pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dict_clustermembers.items()]))
