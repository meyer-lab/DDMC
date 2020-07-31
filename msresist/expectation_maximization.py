"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from .gmm import gmm_initialize, m_step
from .binomial import assignPeptidesBN, GenerateBPM, BackgroundSeqs, position_weight_matrix
from .pam250 import assignPeptidesPAM, MotifPam250Scores
from .motifs import ForegroundSeqs


def EM_clustering_opt(data, info, ncl, SeqWeight, distance_method, max_n_iter, n_runs):
    """ Run Coclustering n times and return the best fit. """
    scores, products = [], []
    for _ in range(n_runs):
        cl_seqs, labels, score, n_iter, gmmp, wins = EM_clustering(
            data, info, ncl, SeqWeight, distance_method, max_n_iter
        )
        scores.append(score)
        products.append([cl_seqs, labels, score, n_iter, gmmp, wins])

    idx = np.argmax(scores)
    return products[idx][0], products[idx][1], products[idx][2], products[idx][3], products[idx][4], products[idx][5]


def EM_clustering(data, info, ncl, SeqWeight, distance_method, max_n_iter):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    X = pd.concat([info, data.T], axis=1)
    d = np.array(data.T)
    sequences = ForegroundSeqs(list(X["Sequence"]))

    # Initialize model
    gmm, cl_seqs, gmmp, labels = gmm_initialize(X, ncl)
    background = GenerateSeqBackgroundAndPAMscores(X["Sequence"], distance_method)

    # EM algorithm
    store_labels = []
    for n_iter in range(max_n_iter):
        # E step: Assignment of each peptide based on data and seq
        if distance_method == "Binomial":
            binoM = GenerateBPM(cl_seqs, background)
            seq_scores = assignPeptidesBN(ncl, sequences, binoM)
        else:
            seq_scores = assignPeptidesPAM(ncl, cl_seqs, background, labels)

        final_scores = seq_scores * SeqWeight + gmmp
        SeqIdx = np.argmax(seq_scores, axis=1)
        labels = np.argmax(final_scores, axis=1)
        DataIdx = np.argmax(gmmp, axis=1)
        scores = np.max(final_scores, axis=1)

        assert np.all(np.isfinite(scores)), \
        f"Final scores not finite, seq_scores = {seq_scores}, gmmp = {gmmp}"

        cl_seqs = [[] for i in range(ncl)]
        for j, motif in enumerate(sequences):
            cl_seqs[labels[j]].append(motif)

        # Count wins
        SeqWins = np.sum((SeqIdx == labels) & (DataIdx != labels))
        DataWins = np.sum((DataIdx == labels) & (SeqIdx != labels))
        BothWin = np.sum((DataIdx == labels) & (SeqIdx == labels))
        MixWins = np.sum((DataIdx != labels) & (SeqIdx != labels))

        # Assert there are at least three peptides per cluster, otherwise re-initialize algorithm
        if True in [len(sl) < 3 for sl in cl_seqs]:
            print(f"Re-initialize GMM clusters, empty cluster(s) at iteration {n_iter}")
            gmm, cl_seqs, gmmp, labels = gmm_initialize(X, ncl)
            store_labels = []

        # Store current results
        store_labels.append(labels)
        wins = (SeqWins, DataWins, BothWin, MixWins)

        # M step: Update motifs, cluster centers, and gmm probabilities
        m_step(d, gmm, HardAssignments(labels, ncl))
        gmmp = gmm.predict_proba(d)

        if True in np.isnan(gmmp):
            print(f"Re-initialize GMM, NaN responsibilities at iteration {n_iter}")
            gmm, cl_seqs, gmmp, labels = gmm_initialize(X, ncl)
            store_labels = []

        if len(store_labels) > 4:
            # Check convergence
            converge = False
            for i in range(4):
                if adjusted_rand_score(store_labels[-1], store_labels[-i]) == 1:
                    converge = True
                    break

            if converge:
                cl_seqs = [[str(seq) for seq in cluster] for cluster in cl_seqs]
                return cl_seqs, labels, np.mean(scores), n_iter, gmm, wins

    print("convergence has not been reached. Clusters: %s SeqWeight: %s" % (ncl, SeqWeight))
    cl_seqs = [[str(seq) for seq in cluster] for cluster in cl_seqs]
    return cl_seqs, np.array(labels), np.mean(scores), n_iter, gmm, wins


def HardAssignments(labels, ncl):
    """ Generate a responsibility matrix with hard assignments, 
    i.e. 1 for assignments, 0 otherwise. """
    m = np.zeros((len(labels), ncl))

    for ii, idx in enumerate(labels):
        m[ii, idx] = 1.0

    assert np.all(np.sum(m, axis=0) >= 1.0)
    assert np.all(np.sum(m, axis=1) == 1.0)

    return m


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
