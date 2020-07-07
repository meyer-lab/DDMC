"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import numpy as np
import pandas as pd
from msresist.gmm import gmm_initialize, m_step
from msresist.binomial import GenerateBPM, TranslateMotifsToIdx, MeanBinomProbs, BackgroundSeqs, position_weight_matrix
from msresist.pam250 import MotifPam250Scores, pairwise_score
from msresist.motifs import ForegroundSeqs, CountPsiteTypes


def EM_clustering_opt(data, info, ncl, SeqWeight, distance_method, max_n_iter, n_runs):
    """ Run Coclustering n times and return the best fit. """
    scores, products = [], []
    for _ in range(n_runs):
        cl_seqs, labels, score, n_iter, gmmp, wins = EM_clustering(data, info, ncl, SeqWeight, distance_method, max_n_iter)
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
    gmm, cl_seqs, gmmp, new_labels = gmm_initialize(X, ncl, distance_method)
    bg_pwm, Seq1Seq2ToScores = GenerateSeqBackgroundAndPAMscores(X["Sequence"], distance_method)
    

    # EM algorithm
    store_Clseqs = []
    for n_iter in range(max_n_iter):
        labels, scores, wins = [], [], []
        seq_reassign = [[] for i in range(ncl)]

        # E step: Assignment of each peptide based on data and seq
        if distance_method == "Binomial":
            binoM = GenerateBPM(cl_seqs, bg_pwm)
        else:
            binoM = None

        SeqWins, DataWins, BothWin, MixWins = 0, 0, 0, 0
        for j, motif in enumerate(sequences):
            score, idx, SeqIdx, DataIdx = assignPeptides(
                ncl, motif, distance_method, SeqWeight, gmmp, j, bg_pwm, cl_seqs, binoM, Seq1Seq2ToScores, new_labels
            )
            labels.append(idx)
            scores.append(score)
            seq_reassign[idx].append(motif)
            SeqWins, DataWins, BothWin, MixWins = TrackWins(idx, SeqIdx, DataIdx, SeqWins, DataWins, BothWin, MixWins)

        # Assert there are at least three peptides per cluster, otherwise re-initialize algorithm
        if True in [len(sl) < 3 for sl in seq_reassign]:
            print("Re-initialize GMM clusters, empty cluster(s) at iteration %s" % (n_iter))
            gmm, cl_seqs, gmmp, new_labels = gmm_initialize(X, ncl, distance_method)
            assert cl_seqs != seq_reassign, "Same cluster assignments after re-initialization"
            assert [len(sublist) > 0 for sublist in cl_seqs], "Empty cluster(s) after re-initialization"
            store_Clseqs = []
            continue

        # Store current results
        store_Clseqs.append(cl_seqs)
        new_score = np.mean(scores)
        new_labels = np.array(labels)
        wins = "SeqWins: " + str(SeqWins) + " DataWins: " + str(DataWins) + " BothWin: " + str(BothWin) + " MixWin: " + str(MixWins)

        # M step: Update motifs, cluster centers, and gmm probabilities
        cl_seqs = seq_reassign
        gmmp_hard = HardAssignments(new_labels, ncl)
        m_step(d, gmm, gmmp_hard)
        gmmp = gmm.predict_proba(d)

        if True in np.isnan(gmmp):
            print("Re-initialize GMM, NaN responsibilities at iteration %s" % (n_iter))
            gmm, cl_seqs, gmmp, new_labels = gmm_initialize(X, ncl, distance_method)
            assert cl_seqs != seq_reassign, "Same cluster assignments after re-initialization"
            assert [len(sublist) > 0 for sublist in cl_seqs], "Empty cluster(s) after re-initialization"
            store_Clseqs = []
            continue

        if len(store_Clseqs) > 2:
            # Check convergence
            if store_Clseqs[-1] == store_Clseqs[-2] or store_Clseqs[-1] == store_Clseqs[-3]:
                cl_seqs = [[str(seq) for seq in cluster] for cluster in cl_seqs]
                return cl_seqs, new_labels, new_score, n_iter, gmm, wins

    print("convergence has not been reached. Clusters: %s SeqWeight: %s" % (ncl, SeqWeight))
    cl_seqs = [[str(seq) for seq in cluster] for cluster in cl_seqs]
    return cl_seqs, np.array(labels), np.mean(scores), n_iter, gmm, wins


def assignPeptides(ncl, motif, distance_method, SeqWeight, gmmp, j, bg_pwm, cl_seqs, binomials, Seq1Seq2ToScore, labels):
    """E-step––Do the peptide assignment according to sequence and data"""
    data_scores = np.zeros(ncl,)
    seq_scores = np.zeros(ncl,)
    final_scores = np.zeros(ncl,)
    # Binomial Probability Matrix distance (p-values) between foreground and background sequences
    if distance_method == "Binomial":
        NumMotif = TranslateMotifsToIdx(motif)

        for z in range(ncl):
            BPM_score = MeanBinomProbs(binomials[z], NumMotif)
            seq_scores[z] = BPM_score
            data_scores[z] = gmmp[j, z]
            final_scores[z] = BPM_score * SeqWeight + gmmp[j, z]

    # Average distance between each sequence and any cluster based on PAM250 substitution matrix
    if distance_method == "PAM250":
        seq_scores = np.zeros(ncl,)
        for idx, assignments in enumerate(labels):
            seq_scores[assignments] += Seq1Seq2ToScore[j, idx]

        for z in range(ncl):
            seq_scores[z] /= len(cl_seqs[z])  # average score per cluster
            data_scores[z] = gmmp[j, z]
            final_scores[z] = seq_scores[z] * SeqWeight + gmmp[j, z]

    DataIdx = np.argmax(data_scores)
    SeqIdx = np.argmax(seq_scores)
    idx = np.argmax(final_scores)

    score = final_scores[idx]
    assert math.isnan(score) == False and math.isinf(score) == False, \
        f"final score is either NaN or -Inf, motif = {motif}, gmmp = {gmmp}, nonzeros = {np.count_nonzero(gmmp)}"

    return score, idx, SeqIdx, DataIdx


def TrackWins(idx, SeqIdx, DataIdx, SeqWins, DataWins, BothWin, MixWins):
    """ Assess if the finala scaled score was determined by data or sequence """
    if SeqIdx == idx and SeqIdx != DataIdx:
        SeqWins += 1
    elif DataIdx == idx and DataIdx != SeqIdx:
        DataWins += 1
    elif DataIdx == idx and SeqIdx == idx:
        BothWin += 1
    else:
        MixWins += 1
    return SeqWins, DataWins, BothWin, MixWins


def HardAssignments(labels, ncl):
    """ Generate a responsibility matrix with hard assignments, i.e. 1 for assignments, 0 otherwise. """
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
        Seq1Seq2ToScores = False

    elif distance_method == "PAM250":
        # Compute all pairwise distances and generate seq vs seq to score dictionary
        seqs = [s.upper() for s in sequences]
        Seq1Seq2ToScores = MotifPam250Scores(seqs)
        bg_pwm = False

    return bg_pwm, Seq1Seq2ToScores
