"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from .gmm import gmm_initialize, m_step
from .binomial import assignPeptidesBN, GenerateBPM, BackgroundSeqs, position_weight_matrix
from .pam250 import assignPeptidesPAM, MotifPam250Scores
from .motifs import ForegroundSeqs


def EM_clustering_opt(data, info, ncl, SeqWeight, distance_method, max_n_iter, n_runs, background):
    """ Run Coclustering n times and return the best fit. """
    scores, products = [], []
    for _ in range(n_runs):
        score = EM_clustering(data, info, ncl, SeqWeight, distance_method, max_n_iter, background)
        scores.append(score)
        #products.append([converge, cl_seqs, labels, score, n_iter, gmmp, wins])

    idx = np.argmax(scores)
    return scores


def EM_clustering(data, info, ncl, SeqWeight, distance_method, max_n_iter, background):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    X = pd.concat([info, data.T], axis=1)
    d = np.array(data.T)
    sequences = ForegroundSeqs(list(X["Sequence"]))

    # Initialize model
    gmm, gmmp = gmm_initialize(X, ncl)
    scores = gmmp

    if type(background) == bool:
        background = GenerateSeqBackgroundAndPAMscores(X["Sequence"], distance_method)

    # EM algorithm
    for n_iter in range(max_n_iter):
        # E step: Assignment of each peptide based on data and seq
        if distance_method == "Binomial":
            binoM = GenerateBPM(scores, background)
            seq_scores = assignPeptidesBN(ncl, sequences, binoM)
        else:
            seq_scores = assignPeptidesPAM(ncl, scores, background)

        scores = seq_scores * SeqWeight + gmmp
        # Probabilities should sum to one across clusters
        scores /= np.sum(scores, axis=1)[:, np.newaxis]

        assert np.all(np.isfinite(scores)), \
            f"Final scores not finite, seq_scores = {seq_scores}, gmmp = {gmmp}"

        # M step: Update motifs, cluster centers, and gmm probabilities
        m_step(d, gmm, scores)
        gmmp = gmm.predict_proba(d)

        assert np.all(np.isfinite(gmmp)), \
            f"gmmp not finite, seq_scores = {seq_scores}, gmmp = {gmmp}"

        if n_iter > 3 and np.linalg.norm(final_scores_last - scores) < 0.0001:
            print("Converged.")
            print(scores)
            return scores

        final_scores_last = np.copy(scores)

    print(f"convergence has not been reached. Clusters: {ncl} SeqWeight: {SeqWeight}")
    return scores


def GenerateSeqBackgroundAndPAMscores(sequences, distance_method):
    if distance_method == "Binomial":
        # Background sequences
        bg_seqs = BackgroundSeqs(sequences)
        bg_pwm = position_weight_matrix(bg_seqs)

    elif distance_method == "PAM250":
        # Compute all pairwise distances and generate seq vs seq to score dictionary
        seqs = [s.upper() for s in sequences]
        bg_pwm = MotifPam250Scores(seqs)
    return bg_pwm