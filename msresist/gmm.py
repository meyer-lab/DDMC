"""Guassian Mixture Model functions to determine cluster assignments based on phosphorylation levels."""

import numpy as np
from pomegranate import GeneralMixtureModel, NormalDistribution
from msresist.motifs import ForegroundSeqs


def gmm_initialize(X, ncl):
    """ Return peptides data set including its labels and pvalues matrix. """
    d = X.select_dtypes(include=["float64"])
    labels, gmmp = [0, 0, 0], [np.nan]

    while len(set(labels)) < ncl or True in np.isnan(gmmp):
        gmm = GeneralMixtureModel.from_samples(NormalDistribution, X=d, n_components=ncl, max_iterations=1)
        labels = gmm.predict(d)
        gmmp = gmm.predict_proba(d)

    X["GMM_cluster"] = labels
    init_clusters = [ForegroundSeqs(list(X[X["GMM_cluster"] == i]["Sequence"])) for i in range(ncl)]

    assert [len(sublist) > 0 for sublist in init_clusters], "Empty cluster(s) on initialization"
    return gmm, init_clusters, gmmp, labels


def m_step(d, gmm, gmmp_hard):
    """ Bypass gmm fitting step by working directly with the distribution objects. """
    for i in range(gmmp_hard.shape[1]):
        weights = gmmp_hard[:, i]
        gmm.distributions[i].fit(d, weights=weights)
