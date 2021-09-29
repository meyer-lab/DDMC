"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob, _estimate_gaussian_parameters, _compute_precision_cholesky
from .soft_impute import SoftImpute


class DDMC(GaussianMixture):
    """ The core DDMC class. """
    def _estimate_log_prob(self, X):
        logp = _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type
        )

        # Add in the sequence effect
        logp += self.seqDist.logWeights

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
        self.seqDist.from_summaries(np.exp(log_resp))


def EM_clustering(data, _, ncl, seqDist=None):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    d = np.array(data.T)

    imp = SoftImpute(J=10, verbose=True)
    imp.fit(d)
    d[np.isnan(d)] = imp.predict(d)
    assert np.all(np.isfinite(d))

    gmm = DDMC(n_components=ncl, covariance_type="diag", n_init=2, verbose=10)
    gmm.seqDist = seqDist

    gmm.fit(d)
    scores = gmm.predict_proba(d)

    seq_scores = scores # fix
    avgScore = gmm.lower_bound_

    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(seq_scores))

    return avgScore, scores, seq_scores, gmm
