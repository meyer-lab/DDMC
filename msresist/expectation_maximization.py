"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

import numpy as np
import pandas as pd
from .gmm import gmm_initialize, m_step
from .binomial import assignPeptidesBN, BackgroundSeqs, position_weight_matrix, GenerateBinarySeqID, AAlist
from .pam250 import assignPeptidesPAM, MotifPam250Scores


def EM_clustering(data, info, ncl, SeqWeight, distance_method, background, bg_mat, dataTensor, max_n_iter=2000):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    X = pd.concat([info, data.T], axis=1)
    d = np.array(data.T)

    # Initialize model
    gmm, gmmp = gmm_initialize(X, ncl)
    scores = gmmp

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
        # E step: Assignment of each peptide based on data and seq
        if distance_method == "Binomial":
            seq_scores = assignPeptidesBN(dataTensor, scores, bg_mat)
        else:
            seq_scores = assignPeptidesPAM(ncl, scores, background)

        # seq_scores is log-likelihood, logaddexp to avoid roundoff error
        scores = np.logaddexp(seq_scores * SeqWeight, np.log(gmmp))
        scores = np.exp(scores)

        # Probabilities should sum to one across clusters
        scores /= np.sum(scores, axis=1)[:, np.newaxis]

        assert np.all(np.isfinite(scores)), \
            f"Final scores not finite, seq_scores = {seq_scores}, gmmp = {gmmp}"

        # M step: Update motifs, cluster centers, and gmm probabilities
        m_step(d, gmm, scores)
        gmmp = gmm.predict_proba(d)

        assert np.all(np.isfinite(gmmp)), \
            f"gmmp not finite, seq_scores = {seq_scores}, gmmp = {gmmp}"

        if n_iter > 3 and np.linalg.norm(final_scores_last - scores) < 1e-8:
            return scores, seq_scores, gmm

        final_scores_last = np.copy(scores)

    print(f"convergence has not been reached. Clusters: {ncl} SeqWeight: {SeqWeight}")
    return scores, seq_scores, gmm
