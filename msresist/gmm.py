"""Guassian Mixture Model functions to determine cluster assignments based on phosphorylation levels."""

import numpy as np
from pomegranate import GeneralMixtureModel, NormalDistribution


def gmm_initialize(X, ncl):
    """ Return peptides data set including its labels and pvalues matrix. """
    d = X.select_dtypes(include=["float64"])
    gmmp = [np.nan]

    tries = 0
    while np.any(np.isnan(gmmp)):
        gmm = GeneralMixtureModel.from_samples(NormalDistribution, X=d, n_components=ncl, max_iterations=1)
        gmmp = gmm.predict_proba(d)
        tries += 1
        if tries == 300:
            raise RuntimeError("GMM can't fit, try a smaller number of clusters.")

    return gmm, gmmp


def m_step(d, gmm, gmmp_hard):
    """ Bypass gmm fitting step by working directly with the distribution objects. """
    for i in range(gmmp_hard.shape[1]):
        weights = gmmp_hard[:, i]
        gmm.distributions[i].fit(d, weights=weights)
