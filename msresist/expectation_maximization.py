"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

import numpy as np
from statsmodels.multivariate.pca import PCA
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob, _estimate_gaussian_parameters, _compute_precision_cholesky


class DDMC(GaussianMixture):
    """ The core DDMC class. """
    def _estimate_log_prob(self, X):
        logp = _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        )

        # Add in the sequence effect
        for ii in range(self.n_components):
            logp[:, ii] += self.seqDist[ii].logWeights

        return logp

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

        # Add in the sequence effect
        for ii in range(self.n_components):
            self.seqDist[ii].from_summaries(np.squeeze(np.exp(log_resp[:, ii])))


def EM_clustering(data, _, ncl, seqDist=None):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    d = np.array(data.T)

    pc = PCA(d, ncomp=5, method="nipals", missing="fill-em", standardize=False, demean=False, normalize=False)
    d = pc._adjusted_data
    assert np.all(np.isfinite(d))

    gmm = DDMC(n_components=ncl, covariance_type="diag", n_init=5)

    if isinstance(seqDist, list):
        gmm.seqDist = seqDist
    else:
        gmm.seqDist = [seqDist.copy() for _ in range(ncl)]

    gmm.fit(d)
    scores = gmm.predict_proba(d)

    seq_scores = scores # fix
    avgScore = gmm.lower_bound_

    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(seq_scores))

    return avgScore, scores, seq_scores, gmm
