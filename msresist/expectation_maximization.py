"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

import numpy as np
import pandas as pd
import scipy.stats as sp
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score
from pomegranate import GeneralMixtureModel, NormalDistribution
from .binomial import assignPeptidesBN, BackgroundSeqs, position_weight_matrix, GenerateBinarySeqID, AAlist
from .pam250 import assignPeptidesPAM, MotifPam250Scores
from .motifs import ForegroundSeqs


def EM_clustering(data, info, ncl, SeqWeight, distance_method, background, max_n_iter=10000):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    X = pd.concat([info, data.T], axis=1)
    d = np.array(data.T)

    # Initialize model
    gmm = GeneralMixtureModel.from_samples(NormalDistribution, X=X.select_dtypes(include=["float64"]), n_components=ncl)
    scores = np.log(sp.dirichlet.rvs(np.ones(ncl) / ncl, size=X.shape[0]))

    if type(background) == bool:
        seqs = [s.upper() for s in X["Sequence"]]

        if distance_method == "Binomial":
            # Background sequences
            background = position_weight_matrix(BackgroundSeqs(X["Sequence"]))
            bg_mat = np.array([background[AA] for AA in AAlist])
            dataTensor = GenerateBinarySeqID(seqs)

        elif distance_method == "PAM250":
            # Compute all pairwise distances and generate seq vs seq to score dictionary
            background = MotifPam250Scores(seqs)

    # EM algorithm
    for n_iter in range(max_n_iter):
        # M step: Update motifs, cluster centers, and gmm probabilities
        for i in range(ncl):
            gmm.distributions[i].fit(d, weights=scores[:, i])

        gmmp = gmm.predict_log_proba(d)
        assert np.all(np.isfinite(gmmp)), f"gmmp = {gmmp}, niter = {n_iter}"

        # E step: Assignment of each peptide based on data and seq
        if distance_method == "Binomial":
            seq_scores = assignPeptidesBN(dataTensor, np.exp(scores), bg_mat)
        else:
            seq_scores = assignPeptidesPAM(ncl, np.exp(scores), background)
        assert np.all(np.isfinite(seq_scores)), f"seq_scores = {seq_scores}, gmmp = {gmmp}"

        # seq_scores is log-likelihood, logaddexp to avoid roundoff error
        seq_scores -= logsumexp(seq_scores, axis=1)[:, np.newaxis]
        scores = np.logaddexp(seq_scores * SeqWeight, gmmp)

        # Probabilities should sum to one across clusters
        scores -= logsumexp(scores, axis=1)[:, np.newaxis]
        assert np.all(np.isfinite(scores)), f"seq_scores = {seq_scores}, gmmp = {gmmp}"

        if n_iter > 2 and np.linalg.norm(final_scores_last - scores) < 1e-6:
            return scores, seq_scores, gmm

        final_scores_last = np.copy(scores)

    print(f"Did not reach convergence. Clusters: {ncl} SeqWeight: {SeqWeight}")
    return scores, seq_scores, gmm
