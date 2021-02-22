"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

import numpy as np
from sklearn.cluster import KMeans
from statsmodels.multivariate.pca import PCA
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

    # In case we have missing data, use SVD-EM to fill it for initialization
    pc = PCA(d, ncomp=ncl, missing="fill-em", standardize=False, demean=False, normalize=False)

    # Solve for the KMeans clustering for initialization
    km = KMeans(ncl, tol=1e-9)
    km.fit(pc._adjusted_data)

    # Add a dummy variable for the sequence information
    d = np.hstack((d, idxx.T))

    for _ in range(10):
        if gmmIn is None:
            # Initialize model
            dists = list()
            for ii in range(ncl):
                nDist = [NormalDistribution(1.0, 0.2) for _ in range(d.shape[1] - 1)] + [seqDist.copy()]

                for jj in range(d.shape[1] - 1):
                    nDist[jj].fit(d[km.labels_ == ii, jj])

                weights = np.array(km.labels_ == ii, dtype=float)
                weights = 0.9 * weights + 0.01

                nDist[-1].summarize(d[:, -1], weights=weights)
                nDist[-1].from_summaries()

                dists.append(IndependentComponentsDistribution(nDist))

            gmm = GeneralMixtureModel(dists)
        else:
            gmm = gmmIn

        gmm.fit(d, max_iterations=200, verbose=False, stop_threshold=1e-3)
        scores = gmm.predict_proba(d)

        if np.all(np.isfinite(scores)):
            break

    seq_scores = np.exp([dd[-1].logWeights for dd in gmm.distributions])
    avgScore = np.sum(gmm.log_probability(d))

    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(seq_scores))

    return avgScore, scores, seq_scores, gmm
