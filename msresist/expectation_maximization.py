"""Main Expectation-Maximization functions using gmm and binomial or pam250 to determine cluster assginments.
EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix """

import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import numpy as np
import pandas as pd
from msresist.gmm import gmm_initialize, m_step, GmmpCompatibleWithSeqScores
from msresist.binomial import GenerateBPM, TranslateMotifsToIdx, MeanBinomProbs, BackgroundSeqs, position_weight_matrix
from msresist.pam250 import MotifPam250Scores, pairwise_score
from msresist.motifs import ForegroundSeqs, CountPsiteTypes


def EM_clustering_opt(data, info, ncl, SeqWeight, distance_method, gmm_method, max_n_iter, n_runs):
    """ Run Coclustering n times and return the best fit. """
    scores, products = [], []
    for _ in range(n_runs):
        cl_seqs, labels, score, n_iter, gmmp, wins = EM_clustering(data, info, ncl, SeqWeight,
                                                       distance_method, gmm_method, max_n_iter)
        scores.append(score)
        products.append([cl_seqs, labels, score, n_iter, gmmp, wins])

    if distance_method == "Binomial":
        idx = np.argmin(scores)
    elif distance_method == "PAM250":
        idx = np.argmax(scores)
    return products[idx][0], products[idx][1], products[idx][2], products[idx][3], products[idx][4], products[idx][5]


def EM_clustering(data, info, ncl, SeqWeight, distance_method, gmm_method, max_n_iter):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    ABC = pd.concat([info, data.T], axis=1)
    d = np.array(data.T)
    sequences = ForegroundSeqs(list(ABC["Sequence"]))

    # Initialize with gmm clusters and generate gmm pval matrix
    gmm, cl_seqs, gmmp = gmm_initialize(ABC, ncl, distance_method, gmm_method)

    # Background sequences
    if distance_method == "Binomial":
        bg_seqs = BackgroundSeqs(ABC)
        bg_pwm = position_weight_matrix(bg_seqs)
        Seq1Seq2ToScores = False

    elif distance_method == "PAM250":
        # Compute all pairwsie distances and generate seq vs seq to score dictionary
        seqs = [s.upper() for s in ABC["Sequence"]]
        Seq1Seq2ToScores = MotifPam250Scores(seqs)
        bg_pwm = False

    # EM algorithm
    store_Clseqs, store_scores, store_labels = [], [], []
    store_labels.append(gmm.predict(d))
    for n_iter in range(max_n_iter):
        labels, scores, wins = [], [], []
        seq_reassign = [[] for i in range(ncl)]

        # E step: Assignment of each peptide based on data and seq
        binoM = GenerateBPM(cl_seqs, distance_method, bg_pwm)
        SeqWins, DataWins, BothWin, MixWins = 0, 0, 0, 0
        for j, motif in enumerate(sequences):
            score, idx, SeqIdx, DataIdx = assignSeqs(ncl, motif, distance_method, SeqWeight, gmmp,
                                                     j, bg_pwm, cl_seqs, binoM, Seq1Seq2ToScores,
                                                     store_labels[-1])
            labels.append(idx)
            scores.append(score)
            seq_reassign[idx].append(motif)
            SeqWins, DataWins, BothWin, MixWins = TrackWins(idx, SeqIdx, DataIdx, 
                                                            SeqWins, DataWins, BothWin, MixWins)

        # Assert there are at least two peptides per cluster, otherwise re-initialize algorithm
        if True in [len(sl) < 1 for sl in seq_reassign]:
            print("Re-initialize GMM clusters, empty cluster(s) at iteration %s" % (n_iter))
            gmm, cl_seqs, gmmp = gmm_initialize(ABC, ncl, distance_method, gmm_method)
            assert cl_seqs != seq_reassign, "Same cluster assignments after re-initialization"
            assert [len(sublist) > 0 for sublist in cl_seqs], \
            "Empty cluster(s) after re-initialization"
            store_Clseqs, store_scores = [], []
            continue

        # Store current results
        store_Clseqs.append(cl_seqs)
        store_scores.append(np.mean(scores))
        store_labels.append(labels)
        wins = "SeqWins: " + str(SeqWins) + " DataWins: " + str(DataWins) + \
            " BothWin: " + str(BothWin) + " MixWin: " + str(MixWins)

        # M step: Update motifs, cluster centers, and gmm probabilities
        cl_seqs = seq_reassign
        gmmp_hard = HardAssignments(labels, ncl)
        m_step(d, gmm, gmmp_hard, gmm_method)
        gmmp = gmm.predict_proba(d)
        gmmp = GmmpCompatibleWithSeqScores(gmmp, distance_method)

        if len(store_scores) > 2:
            # Check convergence
            if store_Clseqs[-1] == store_Clseqs[-2]:
                cl_seqs = [[str(seq) for seq in cluster] for cluster in cl_seqs]
                return cl_seqs, np.array(labels), np.mean(scores), n_iter, gmmp, wins

    print("convergence has not been reached. Clusters: %s SeqWeight: %s" % (ncl, SeqWeight))
    cl_seqs = [[str(seq) for seq in cluster] for cluster in cl_seqs]
    return cl_seqs, np.array(labels), np.mean(scores), n_iter, gmmp, wins


def assignSeqs(ncl, motif, distance_method, SeqWeight, gmmp, j, bg_pwm,
               cl_seqs, binomials, Seq1Seq2ToScore, labels):
    """ Do the sequence assignment. """
    data_scores = np.zeros(ncl,)
    seq_scores = np.zeros(ncl,)
    final_scores = np.zeros(ncl,)
    # Binomial Probability Matrix distance (p-values) between foreground and background sequences
    if distance_method == "Binomial":
        for z in range(ncl):
            NumMotif = TranslateMotifsToIdx(motif, list(bg_pwm.keys()))
            BPM_score = MeanBinomProbs(binomials[z], NumMotif)
            seq_scores[z] = BPM_score
            data_scores[z] = gmmp[j, z]
            final_scores[z] = BPM_score * SeqWeight + gmmp[j, z]
        DataIdx = np.argmin(data_scores)
        SeqIdx = np.argmin(seq_scores)
        idx = np.argmin(final_scores)

    # Average distance between each sequence and any cluster based on PAM250 substitution matrix
    if distance_method == "PAM250":
        seq_scores = np.zeros(ncl, dtype=int)
        for idx, assignments in enumerate(labels):
            seq_scores[assignments] += Seq1Seq2ToScore[j, idx]

        for z in range(ncl):
            #             seq_scores[z] = Seq1Seq2ToScore[Seq1Seq2ToScore[:, 0] == z][:, j+1].sum()
            seq_scores[z] /= len(cl_seqs[z])  # average score per cluster
            data_scores[z] = gmmp[j, z]
            final_scores[z] = seq_scores[z] * SeqWeight + gmmp[j, z]
        DataIdx = np.argmax(data_scores)
        SeqIdx = np.argmax(seq_scores)
        idx = np.argmax(final_scores)

    score = final_scores[idx]
    assert math.isnan(score) == False and math.isinf(score) == False, "final score is either \
    NaN or -Inf, motif = %s, gmmp = %s, nonzeros = %s" % (motif, gmmp, np.count_nonzero(gmmp))

    return score, idx, SeqIdx, DataIdx


def e_step(X, cl_seqs, gmmp, distance_method, SeqWeight, ncl):
    """ Expectation step of the EM algorithm. Used for predict and score in
    clustering.py """
    sequences = ForegroundSeqs(X["Sequence"])
    cl_seqs = [ForegroundSeqs(cl) for cl in cl_seqs]

    if distance_method == "Binomial":
        bg_seqs = BackgroundSeqs(X)
        bg_pwm = position_weight_matrix(bg_seqs)

    elif distance_method == "PAM250":
        bg_pwm = False

    labels = np.zeros(len(sequences), dtype=int)
    scores = np.zeros(len(sequences), dtype=float)

    binomials = GenerateBPM(cl_seqs, distance_method, bg_pwm)
    for j, motif in enumerate(sequences):
        final_scores = np.zeros(ncl,)
        # Binomial Probability Matrix distance (p-values) between foreground and background sequences
        if distance_method == "Binomial":
            for z in range(ncl):
                NumMotif = TranslateMotifsToIdx(motif, list(bg_pwm.keys()))
                BPM_score = MeanBinomProbs(binomials[z], NumMotif)
                final_scores[z] = gmmp[j, z] + BPM_score * SeqWeight
            idx = np.argmin(final_scores)

        # Average distance between each sequence and any cluster based on PAM250 substitution matrix
        if distance_method == "PAM250":
            for z in range(ncl):
                PAM250_score = 0
                for seq in cl_seqs[z]:
                    PAM250_score += pairwise_score(motif, seq)
                PAM250_score /= len(cl_seqs[z])
                final_scores[z] = gmmp[j, z] + PAM250_score * SeqWeight
            idx = np.argmax(final_scores)

        labels[j] = idx
        scores[j] = final_scores[idx]

    return np.array(labels), np.mean(scores)


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
    m = []
    for idx in labels:
        l = [0] * ncl
        l[idx] = 1.0
        m.append(l)
    return np.array(m)
