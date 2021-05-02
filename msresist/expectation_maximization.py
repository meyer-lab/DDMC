"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

from copy import copy
import numpy as np
import scipy.stats as sp
from pomegranate import GeneralMixtureModel, NormalDistribution, IndependentComponentsDistribution


def EM_clustering_repeat(nRepeats=3, *params):
    output = EM_clustering(*params)

    for _ in range(nRepeats):
        output_temp = EM_clustering(*params)

        # Use the new result if it's better
        if output_temp[0] > output[0]:
            output = output_temp

    return output


def EM_clustering(data, info, ncl, seqDist=None, gmmIn=None):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    d = np.array(data.T)

    # Indices for looking up probabilities later.
    idxx = np.atleast_2d(np.arange(d.shape[0]))
    d = np.hstack((d, idxx.T))

    for _ in range(10):
        if gmmIn is None:
            # Initialize model
            dists = list()
            for ii in range(ncl):
                nDist = [NormalDistribution(sp.norm.rvs(), 0.2) for _ in range(d.shape[1] - 1)]

                if isinstance(seqDist, list):
                    nDist.append(seqDist[ii])
                else:
                    nDist.append(seqDist.copy())

                dists.append(IndependentComponentsDistribution(nDist))

            gmm = GeneralMixtureModel(dists)
        else:
            gmm = gmmIn

        gmm.fit(d, max_iterations=500, verbose=True, stop_threshold=1e-6)
        scores = gmm.predict_proba(d)

        if np.all(np.isfinite(scores)):
            break

    seq_scores = np.exp([dd[-1].logWeights for dd in gmm.distributions])
    avgScore = np.sum(gmm.log_probability(d))

    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(seq_scores))

    return avgScore, scores, seq_scores, gmm
