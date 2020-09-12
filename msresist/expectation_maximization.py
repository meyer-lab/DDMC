"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

import numpy as np
import scipy.stats as sp
from pomegranate import GeneralMixtureModel, NormalDistribution
from .binomial import assignPeptidesBN, BackgroundSeqs, position_weight_matrix, GenerateBinarySeqID, AAlist
from .pam250 import assignPeptidesPAM, MotifPam250Scores


def EM_clustering(data, info, ncl, SeqWeight, distance_method, background, bg_mat, dataTensor, max_n_iter=2000):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    d = np.array(data.T)

    # Initialize model
    gmm = GeneralMixtureModel.from_samples(NormalDistribution, X=d, n_components=ncl, max_iterations=10)
    scores = sp.dirichlet.rvs(alpha=np.ones(ncl) / ncl, size=data.shape[1])

    if type(background) == bool:
        seqs = [s.upper() for s in info["Sequence"]]

        if distance_method == "Binomial":
            # Background sequences
            background = position_weight_matrix(BackgroundSeqs(info["Sequence"]))
            bg_mat = np.array([background[AA] for AA in AAlist])
            dataTensor = GenerateBinarySeqID(seqs)

        elif distance_method == "PAM250":
            # Compute all pairwise distances and generate seq vs seq to score dictionary
            background = MotifPam250Scores(seqs)

    # EM algorithm
    for n_iter in range(max_n_iter):
        # M step: Update motifs, cluster centers, and gmm probabilities
        if distance_method == "Binomial":
            seq_scores = assignPeptidesBN(dataTensor, scores, bg_mat)
        else:
            seq_scores = assignPeptidesPAM(ncl, scores, background)

        # Rescale scores so they aren't extreme
        seq_scores -= np.mean(seq_scores, axis=1)[:, np.newaxis]
        assert np.all(np.isfinite(seq_scores)), f"seq_scores = {seq_scores}, gmmp = {gmmp}"

        # cluster centers
        for i, dist in enumerate(gmm.distributions):
            dist.fit(d, weights=scores[:, i])

        gmmp = gmm.predict_log_proba(d)
        while np.any(np.isnan(gmmp)):
            print("Restarting GMM as gmmp is nan.")
            gmm = GeneralMixtureModel.from_samples(NormalDistribution, X=d, n_components=ncl)
            gmmp = gmm.predict_log_proba(d)

        assert np.all(np.isfinite(gmmp)), f"seq_scores = {seq_scores}, gmmp = {gmmp}, scores = {scores}"

        # seq_scores is log-likelihood, logaddexp to avoid roundoff error
        scores = np.logaddexp(seq_scores * SeqWeight, gmmp)
        scores = np.exp(scores)

        # Probabilities should sum to one across clusters
        scores /= np.sum(scores, axis=1)[:, np.newaxis]
        avgScore = np.average(np.max(scores, axis=1))

        assert np.all(np.isfinite(scores)), f"seq_scores = {seq_scores}, gmmp = {gmmp}"

        if n_iter > 2 and np.linalg.norm(final_scores_last - scores) < 1e-8:
            return avgScore, scores, seq_scores, gmm

        final_scores_last = np.copy(scores)

    print(f"convergence has not been reached. Clusters: {ncl} SeqWeight: {SeqWeight}")
    return avgScore, scores, seq_scores, gmm
