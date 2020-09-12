"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

import numpy as np
import scipy.stats as sp
from pomegranate import GeneralMixtureModel, NormalDistribution, IndependentComponentsDistribution
from .binomial import Binomial
from .pam250 import PAM250


def EM_clustering(data, info, ncl, SeqWeight, distance_method, background, bg_mat, dataTensor):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    d = np.array(data.T)

    # Indices for looking up probabilities later.
    idxx = np.atleast_2d(np.arange(d.shape[0]))
    d = np.hstack((d, idxx.T))

    # Initialize model
    dist = NormalDistribution(sp.norm.rvs(), 1.0)

    if distance_method == "PAM250":
        seqDist = PAM250(info, background, SeqWeight)
    elif distance_method == "Binomial":
        seqDist = Binomial(info, background, SeqWeight)

    idist = IndependentComponentsDistribution([dist] * (d.shape[1] - 1)  +  [seqDist])
    gmm = GeneralMixtureModel([idist] * ncl)

    gmm.fit(d, inertia=0.1, stop_threshold=1e-12)
    scores = gmm.predict_proba(d)

    seq_scores = np.exp([dd[-1].weights for dd in gmm.distributions])
    avgScore = np.sum(gmm.log_probability(d))

    return avgScore, scores, seq_scores, gmm
