"""Guassian Mixture Model functions to determine cluster assignments based on phosphorylation levels."""


import numpy as np
from pomegranate import GeneralMixtureModel, NormalDistribution
from sklearn.mixture import GaussianMixture
from msresist.motifs import ForegroundSeqs


def gmm_initialize(X, ncl, distance_method, gmm_method):
    """ Return peptides data set including its labels and pvalues matrix. """
    d = X.select_dtypes(include=["float64"])
    labels, gmm_pred = [0, 0, 0], [np.nan]

    if gmm_method == "pom":
        while len(set(labels)) < ncl or True in np.isnan(gmm_pred):
            gmm = GeneralMixtureModel.from_samples(NormalDistribution, X=d, n_components=ncl, max_iterations=1)
            labels = gmm.predict(d)
            gmm_pred = gmm.predict_proba(d)

    elif gmm_method == "sklearn":
        gmm = GaussianMixture(n_components=ncl, max_iter=1).fit(d)
        labels = gmm.predict(d)
        gmm_pred = gmm.predict_proba(d)

    gmmp = GmmpCompatibleWithSeqScores(gmm_pred, distance_method)

    X["GMM_cluster"] = labels
    init_clusters = [ForegroundSeqs(list(X[X["GMM_cluster"] == i]["Sequence"])) for i in range(ncl)]
    return gmm, init_clusters, gmmp


def m_step(d, gmm, gmmp_hard, gmm_method):
    """ Bypass gmm fitting step by working directly with the distribution objects. """
    if gmm_method == "pom":
        for i in range(gmmp_hard.shape[1]):
            weights = gmmp_hard[:, i]
            gmm.distributions[i].fit(d, weights=weights)

    if gmm_method == "sklearn":
        gmm._m_step(d, gmmp_hard)


def GmmpCompatibleWithSeqScores(gmm_pred, distance_method):
    """ Make data and sequence scores as close in magnitude as possible. """
    if distance_method == "PAM250":
        gmmp = gmm_pred * 100
    elif distance_method == "Binomial":
        gmm_pred[gmm_pred == 1] = 0.9999999999999
        gmmp = np.log(1 - gmm_pred)
    return gmmp
