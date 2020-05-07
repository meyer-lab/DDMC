"""Sequence Analysis Functions. """

import os
import re
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import numpy as np
import pandas as pd
from Bio import SeqIO, motifs
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.SubsMat import MatrixInfo
from scipy.stats import binom
from pomegranate import GeneralMixtureModel, NormalDistribution

path = os.path.dirname(os.path.abspath(__file__))


###------------ Mapping to Uniprot's Proteome To Generate +/-5AA p-site Motifs ------------------###

def pYmotifs(X, names):
    " Generate pY motifs for pre-processing. "
    names, seqs, pXpos, Xidx = GeneratingKinaseMotifs(names, FormatSeq(X))
    print(len(pXpos))
    X = X.iloc[Xidx, :]
    X['Gene'] = names
    X['Sequence'] = seqs
    X.insert(3, "Position", pXpos)
    return X[~X["Sequence"].str.contains("-")]


def FormatName(X):
    """ Keep only the general protein name, without any other accession information """
    full = [v.split("OS")[0].strip() for v in X.iloc[:, 0]]
    gene = [v.split("GN=")[1].split(" PE")[0].strip() for v in X.iloc[:, 0]]
    return full, gene


def FormatSeq(X):
    """ Deleting -1/-2 for mapping to uniprot's proteome"""
    return [v.split("-")[0] for v in X["Sequence"]]


def CountPsiteTypes(X, cA):
    """ Count number of different phosphorylation types in a MS data set."""
    pS = 0
    pT = 0
    pY = 0
    primed = 0

    for seq in X:
        if "s" in seq[cA]:
            pS += 1
        if 'y' in seq[cA]:
            pY += 1
        if 't' in seq[cA]:
            pT += 1
        pp = 0
        for i in seq:
            if i.islower():
                pp += 1
        if pp > 1:
            primed += 1

    return pY, pS, pT, primed


def DictProteomeNameToSeq(X, n):
    """ To generate proteom's dictionary """
    DictProtToSeq_UP = {}
    for rec2 in SeqIO.parse(X, "fasta"):
        UP_seq = str(rec2.seq)
        if n == "full":
            UP_name = rec2.description.split("HUMAN ")[1].split(" OS")[0]
            DictProtToSeq_UP[UP_name] = str(UP_seq)
        if n == "gene":
            try:
                UP_name = rec2.description.split(" GN=")[1].split(" ")[0]
                DictProtToSeq_UP[UP_name] = str(UP_seq)
            except BaseException:
                continue
    return DictProtToSeq_UP


def getKeysByValue(dictOfElements, valueToFind):
    """ Find the key of a given value within a dictionary. """
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item in listOfItems:
        if valueToFind in item[1]:
            listOfKeys.append(item[0])
    return listOfKeys


def MatchProtNames(ProteomeDict, MS_names, MS_seqs):
    """ Match protein names of MS and Uniprot's proteome. """
    matchedNames, seqs, Xidx = [], [], []
    counter = 0
    for i, MS_seq in enumerate(MS_seqs):
        MS_seqU = MS_seq.upper()
        MS_name = MS_names[i].strip()
        if MS_name in ProteomeDict and MS_seqU in ProteomeDict[MS_name]:
            Xidx.append(i)
            seqs.append(MS_seq)
            matchedNames.append(MS_name)
        else:
            try:
                newname = getKeysByValue(ProteomeDict, MS_seqU)[0]
                assert MS_seqU in ProteomeDict[newname]
                Xidx.append(i)
                seqs.append(MS_seq)
                matchedNames.append(newname)
            except BaseException:
                counter += 1
                continue

    print(str(counter) + "/" + str(len(MS_seqs)) + " peptides were not found in the proteome.")
    assert len(matchedNames) == len(seqs)
    return matchedNames, seqs, Xidx


def findmotif(MS_seq, MS_name, ProteomeDict, motif_size):
    """ For a given MS peptide, finds it in the ProteomeDict, and maps the +/-5 AA from the p-site, accounting
    for peptides phosphorylated multiple times concurrently. """
    MS_seqU = MS_seq.upper()
    try:
        UP_seq = ProteomeDict[MS_name]
        assert MS_seqU in UP_seq, "check " + MS_name + " with seq " + MS_seq + ". Protein sequence found: " + UP_seq
        regexPattern = re.compile(MS_seqU)
        MatchObs = list(regexPattern.finditer(UP_seq))
        if "y" in MS_seq:
            pY_idx = list(re.compile("y").finditer(MS_seq))
            assert len(pY_idx) != 0
            center_idx = pY_idx[0].start()
            y_idx = center_idx + MatchObs[0].start()
            DoS_idx = None
            if len(pY_idx) > 1:
                DoS_idx = pY_idx[1:]
                assert len(DoS_idx) != 0
            elif "t" in MS_seq or "s" in MS_seq:
                DoS_idx = list(re.compile("y|t|s").finditer(MS_seq))
                assert len(DoS_idx) != 0
            mappedMotif, pidx = makeMotif(UP_seq, MS_seq, motif_size, y_idx, center_idx, DoS_idx)
            if len(pidx) == 1:
                pos = pidx[0]
            if len(pidx) > 1:
                pos = ";".join(pidx)

        if "y" not in MS_seq:
            pTS_idx = list(re.compile("t|s").finditer(MS_seq))
            assert len(pTS_idx) != 0
            center_idx = pTS_idx[0].start()
            ts_idx = center_idx + MatchObs[0].start()
            DoS_idx = None
            if len(pTS_idx) > 1:
                DoS_idx = pTS_idx[1:]
            mappedMotif, pidx = makeMotif(UP_seq, MS_seq, motif_size, ts_idx, center_idx, DoS_idx)
            if len(pidx) == 1:
                pos = pidx[0]
            if len(pidx) > 1:
                pos = ";".join(pidx)

    except BaseException:
        print(MS_name + " not in ProteomeDict.")
        raise

    return pos, mappedMotif


def GeneratingKinaseMotifs(names, seqs):
    """ Main function to generate motifs using 'findmotif'. """
    motif_size = 5
    proteome = open(os.path.join(path, "./data/Sequence_analysis/proteome_uniprot2019.fa"), 'r')
    ProteomeDict = DictProteomeNameToSeq(proteome, n="gene")
    protnames, seqs, Xidx = MatchProtNames(ProteomeDict, names, seqs)
    MS_names, mapped_motifs, uni_pos, = [], [], []

    for i, MS_seq in enumerate(seqs):
        pos, mappedMotif = findmotif(MS_seq, protnames[i], ProteomeDict, motif_size)
        MS_names.append(protnames[i])
        mapped_motifs.append(mappedMotif)
        uni_pos.append(pos)

    proteome.close()
    return MS_names, mapped_motifs, uni_pos, Xidx


def makeMotif(UP_seq, MS_seq, motif_size, ps_protein_idx, center_motif_idx, DoS_idx):
    """ Make a motif out of the matched sequences. """
    UP_seq_copy = list(UP_seq[max(0, ps_protein_idx - motif_size):ps_protein_idx + motif_size + 1])
    assert len(UP_seq_copy) > motif_size, "Size seems too small. " + UP_seq

    # If we ran off the end of the sequence at the beginning or at the end, append a gap
    if ps_protein_idx - motif_size < 0:
        for _ in range(motif_size - ps_protein_idx):
            UP_seq_copy.insert(0, "-")

    elif ps_protein_idx + motif_size + 1 > len(UP_seq):
        for _ in range(ps_protein_idx + motif_size - len(UP_seq) + 1):
            UP_seq_copy.extend("-")

    UP_seq_copy[motif_size] = UP_seq_copy[motif_size].lower()

    pidx = [str(UP_seq_copy[motif_size]).upper() + str(ps_protein_idx + 1) + "-p"]

    # Now go through and copy over phosphorylation
    if DoS_idx:
        for ppIDX in DoS_idx:
            position = ppIDX.start() - center_motif_idx
            # If the phosphosite is within the motif
            if abs(position) < motif_size:
                editPos = position + motif_size
                UP_seq_copy[editPos] = UP_seq_copy[editPos].lower()
                assert UP_seq_copy[editPos] == MS_seq[ppIDX.start()], \
                    UP_seq_copy[editPos] + " " + MS_seq[ppIDX.start()]
                if position != 0:
                    pidx.append(str(UP_seq_copy[editPos]).upper() + str(ps_protein_idx + position + 1) + "-p")

    return ''.join(UP_seq_copy), pidx


###------------ EM Co-Clustering Method using a PAM250 or a Binomial Probability Matrix ------------------###
# Amino acids frequencies (http://www.tiem.utk.edu/~gross/bioed/webmodules/aminoacid.htm) used for pseudocounts,

AAfreq = {"A": 0.074, "R": 0.042, "N": 0.044, "D": 0.059, "C": 0.033, "Q": 0.058, "E": 0.037, "G": 0.074, "H": 0.029, "I": 0.038, "L": 0.076,
          "K": 0.072, "M": 0.018, "F": 0.04, "P": 0.05, "S": 0.081, "T": 0.062, "W": 0.013, "Y": 0.033, "V": 0.068}


def EM_clustering_opt(data, info, ncl, SeqWeight, distance_method, max_n_iter, n_runs):
    """ Run Coclustering n times and return the best fit. """
    scores, products = [], []
    for _ in range(n_runs):
#         print("run: ", i)
        cl_seqs, labels, score, n_iter, gmmp = EM_clustering(data, info, ncl, SeqWeight,
                                                       distance_method, max_n_iter)
        scores.append(score)
        products.append([cl_seqs, labels, score, n_iter, gmmp])

    if distance_method == "Binomial":
        idx = np.argmin(scores)
    elif distance_method == "PAM250":
        idx = np.argmax(scores)

    return products[idx][0], products[idx][1], products[idx][2], products[idx][3], products[idx][4]


def EM_clustering(data, info, ncl, SeqWeight, distance_method, max_n_iter):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    ABC = pd.concat([info, data.T], axis=1)
    d = np.array(data.T)
    sequences = ForegroundSeqs(list(ABC["Sequence"]))

    # Initialize with gmm clusters and generate gmm pval matrix
    print("start initialization...")
    gmm, cl_seqs, gmmp = gmm_initialize(ABC, ncl, distance_method)
    print("gmm initialized")

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
        print("N_ITER: ", n_iter)
        labels, scores = [], []
        seq_reassign = [[] for i in range(ncl)]

        # E step: Assignment of each peptide based on data and seq
        SeqWins, DataWins, BothWin, MixWins = 0, 0, 0, 0
        binoM = GenerateBPM(cl_seqs, distance_method, bg_pwm)
        for j, motif in enumerate(sequences):
            score, idx, SeqIdx, DataIdx = assignSeqs(ncl, motif, distance_method, SeqWeight, gmmp, 
                                    j, bg_pwm, cl_seqs, binoM, Seq1Seq2ToScores, store_labels[-1])
            labels.append(idx)
            scores.append(score)
            seq_reassign[idx].append(motif)
            SeqWins, DataWins, BothWin, MixWins = TrackWins(idx, SeqIdx, DataIdx,
                                                           SeqWins, DataWins, BothWin, MixWins)

        print("SeqW: ", SeqWins, "DataW: ", DataWins, "BothWin: ", BothWin, "MixWins: ", MixWins)
        # Assert there are at least two peptides per cluster, otherwise re-initialize algorithm
        if True in [len(sl) < 2 for sl in seq_reassign]:
            print("Re-initialize GMM clusters, empty cluster(s) at iteration %s" % (n_iter))
            gmm, cl_seqs, gmmp = gmm_initialize(ABC, ncl, distance_method)
            assert cl_seqs != seq_reassign, "Same cluster assignments after re-initialization"
            assert [len(sublist) > 0 for sublist in cl_seqs], "Empty cluster(s) after re-initialization"
            store_Clseqs, store_scores = [], []
            continue

        # Store current results
        store_Clseqs.append(cl_seqs)
        store_scores.append(np.mean(scores))
        store_labels.append(labels)
        print(np.mean(scores))

        # M step: Update motifs, cluster centers, and gmm probabilities
        cl_seqs = seq_reassign
        gmmp = HardAssignments(labels, ncl)
        m_step(d, gmm, gmmp)
        gmmp = gmm.predict_proba(d)
        gmmp = GmmpCompatibleWithSeqScores(gmmp, distance_method)

        if True in np.isnan(gmmp):
            print("bad gmm update")
            gmmp = np.zeros((len(sequences), ncl))

        if len(store_scores) > 2:
            # Check convergence
            if store_Clseqs[-1] == store_Clseqs[-2]:
                cl_seqs = [[str(seq) for seq in cluster] for cluster in cl_seqs]
                return cl_seqs, np.array(labels), np.mean(scores), n_iter, gmmp

    print("convergence has not been reached. Clusters: %s SeqWeight: %s" % (ncl, SeqWeight))
    cl_seqs = [[str(seq) for seq in cluster] for cluster in cl_seqs]
    return cl_seqs, np.array(labels), np.mean(scores), n_iter, gmmp


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
            seq_scores[z] /= len(cl_seqs[z]) #average score per cluster
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

    labels = np.zeros(len(sequences))
    scores = np.zeros(len(sequences))

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

    return np.array(labels), np.array(scores)


###------------ Clustering Phosphorylation Levels with a Gaussian Mixture Model ------------------###

def gmm_initialize(X, ncl, distance_method):
    """ Return peptides data set including its labels and pvalues matrix. """
    d = X.select_dtypes(include=['float64'])
    labels, gmm_pred = [0, 0, 0], [np.nan]

    while len(set(labels)) < ncl or True in np.isnan(gmm_pred):
        gmm = GeneralMixtureModel.from_samples(NormalDistribution, X=d, n_components=ncl, n_jobs=-1, max_iterations=10)
        labels = gmm.predict(d)
        gmm_pred = gmm.predict_proba(d)

    gmmp = GmmpCompatibleWithSeqScores(gmm_pred, distance_method)

    X["GMM_cluster"] = labels
    init_clusters = [ForegroundSeqs(list(X[X["GMM_cluster"] == i]["Sequence"])) for i in range(ncl)]
    return gmm, init_clusters, gmmp


def m_step(d, gmm, gmmp):
    """ Bypass gmm fitting step by working directly with the distribution objects. """
    for i in range(gmmp.shape[1]):
        weights = gmmp[:, i]
        gmm.distributions[i].fit(d, weights=weights)


###------------ Calculating Sequence Distance using a PAM250 transition matrx ------------------###

def MotifPam250Scores(seqs):
    """ Calculate and store all pairwise pam250 distances before starting """
    n = len(seqs)

    out = np.zeros((n, n), dtype=int)
    shm = shared_memory.SharedMemory(create=True, size=out.nbytes)
    out = np.ndarray(out.shape, dtype=out.dtype, buffer=shm.buf)

    with ProcessPoolExecutor(max_workers=32) as e:
        for ii in range(0, n, 500):
            e.submit(innerloop, seqs, ii, 500, shm.name, out.dtype, n)

    out = out.copy()
    shm.unlink()

    i_upper = np.triu_indices(n, k=1)
    out[i_upper] = out.T[i_upper]

    assert out[5, 5] == pairwise_score(seqs[5], seqs[5]), "PAM250 scores array is wrong."
    return out


def innerloop(seqs, ii, endi, shm_name, ddtype, n):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    out = np.ndarray((n, n), dtype=ddtype, buffer=existing_shm.buf)

    for idxx in range(ii, ii + endi):
        for jj in range(idxx + 1):
            out[idxx, jj] = pairwise_score(seqs[idxx], seqs[jj])

    existing_shm.close()


def pairwise_score(seq1: str, seq2: str) -> float:
    """ Compute distance between two kinase motifs. Note this does not account for gaps. """
    score = 0
    for i in range(len(seq1)):
        if (seq1[i], seq2[i]) in MatrixInfo.pam250:
            score += MatrixInfo.pam250[(seq1[i], seq2[i])]
        else:
            score += MatrixInfo.pam250[(seq2[i], seq1[i])]
    return score


###------------ Calculating Sequence Distance using a Binomial Probability Matrix ------------------###
# Binomial method inspired by Schwartz & Gygi's Nature Biotech 2005: doi:10.1038/nbt1146

def GenerateBPM(cl_seqs, distance_method, bg_pwm):
    """ Generate binomial probability matrix for each cluster of sequences """
    if distance_method == "Binomial":
        bpm = []
        for seqs in cl_seqs:
            f = frequencies(seqs)
            bpm.append(BinomialMatrix(len(seqs), f, bg_pwm))
    if distance_method == "PAM250":
        bpm = False
    return bpm


def position_weight_matrix(seqs):
    """ Build PWM of a given set of sequences. """
    m = motifs.create(seqs)
    return m.counts.normalize(pseudocounts=AAfreq)


def InformationContent(seqs):
    """ The mean of the PSSM is particularly important becuase its value is equal to the
    Kullback-Leibler divergence or relative entropy, and is a measure for the information content
    of the motif compared to the background."""
    m = motifs.create(seqs)
    pssm = m.counts.normalize(pseudocounts=AAfreq).log_odds(AAfreq)
    IC = pssm.mean(AAfreq)
    return IC


def frequencies(seqs):
    """ Build counts matrix of a given set of sequences. """
    m = motifs.create(seqs)
    return m.counts


def BinomialMatrix(n, k, p):
    """ Build binomial probability matrix. Note n is the number of sequences,
    k is the counts matrix of the MS data set, p is the pwm of the background. """
    assert list(k.keys()) == ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                              'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    assert list(p.keys()) == list(k.keys())
    BMP = binom.logsf(k=list(k.values()), n=n, p=list(p.values()), loc=0)
    # make the p-value of Y at pos 0 close to 0 to avoid log(0) = -inf
    BMP[BMP == - np.inf] = np.log(float(10**(-323)))
    return BMP


def ExtractMotif(BMP, freqs, pvalCut=10**(-4), occurCut=7):
    """ Identify the most significant residue/position pairs acroos the binomial
    probability matrix meeting a probability and a occurence threshold."""
    motif = list("X" * 11)
    positions = list(BMP.columns[1:])
    AA = list(BMP.iloc[:, 0])
    BMP = BMP.iloc[:, 1:]
    for i in range(len(positions)):
        DoS = BMP.iloc[:, i].min()
        j = BMP[BMP.iloc[:, i] == DoS].index[0]
        aa = AA[j]
        if DoS < pvalCut or DoS == 0.0 and freqs.iloc[j, i] >= occurCut:
            motif[i] = aa
        else:
            motif[i] = "x"

    return ''.join(motif)


def MeanBinomProbs(BPM, motif):
    """ Take the mean of all pvalues corresponding to each motif residue. """
    probs = 0.0
    for i, aa in enumerate(motif):
        probs += BPM[aa, i]
    return probs / len(motif)


def TranslateMotifsToIdx(motif, aa):
    """ Convert amino acid strings into numbers. """
    ResToNum = dict(zip(aa, np.arange(len(aa))))
    NumMotif = []
    for res in list(motif):
        NumMotif.append(ResToNum[res.upper()])
    assert len(NumMotif) == len(motif)
    return NumMotif


def BackgroundSeqs(X):
    """ Build Background data set with the same proportion of pY, pT, and pS motifs as in the foreground set of sequences.
    Note this PsP data set contains 51976 pY, 226131 pS, 81321 pT
    Source: https://www.phosphosite.org/staticDownloads.action -
    Phosphorylation_site_dataset.gz - Last mod: Wed Dec 04 14:56:35 EST 2019
    Cite: Hornbeck PV, Zhang B, Murray B, Kornhauser JM, Latham V, Skrzypek E PhosphoSitePlus, 2014: mutations,
    PTMs and recalibrations. Nucleic Acids Res. 2015 43:D512-20. PMID: 25514926 """
    # Get porportion of psite types in foreground set
    forseqs = list(X["Sequence"])
    forw_pYn, forw_pSn, forw_pTn, _ = CountPsiteTypes(forseqs, 5)
    forw_tot = forw_pYn + forw_pSn + forw_pTn

    pYf = forw_pYn / forw_tot
    pSf = forw_pSn / forw_tot
    pTf = forw_pTn / forw_tot

    # Import backgroun sequences file
    PsP = pd.read_csv("./msresist/data/Sequence_analysis/pX_dataset_PhosphoSitePlus2019.csv")
    PsP = PsP[~PsP["SITE_+/-7_AA"].str.contains("_")]
    PsP = PsP[~PsP["SITE_+/-7_AA"].str.contains("X")]
    refseqs = list(PsP["SITE_+/-7_AA"])
    len_bg = int(len(refseqs))
    backg_pYn, _, _, _ = CountPsiteTypes(refseqs, 7)

    # Make sure there are enough pY peptides to meet proportions
    if backg_pYn >= len_bg * pYf:
        pYn = int(len_bg * pYf)
        pSn = int(len_bg * pSf)
        pTn = int(len_bg * pTf)

    # Not enough pYs, adjust number of peptides based on maximum number of pY peptides
    else:
        tot_p = int(backg_pYn / pYf)
        pYn = backg_pYn
        pSn = int(tot_p * pSf)
        pTn = int(tot_p * pTf)

    # Build background sequences
    bg_seqs = BackgProportions(refseqs, pYn, pSn, pTn)
    return bg_seqs


def BackgProportions(refseqs, pYn, pSn, pTn):
    """ Provided the proportions, add peptides to background set. """
    y_seqs, s_seqs, t_seqs = [], [], []
    pR = ["y", "t", "s"]
    for seq in refseqs:
        if seq[7] not in pR:
            continue

        motif = str(seq)[7 - 5:7 + 6].upper()
        assert len(motif) == 11, \
            "Wrong sequence length. Sliced: %s, Full: %s" % (motif, seq)
        assert motif[5].lower() in pR, \
            "Wrong central AA in background set. Sliced: %s, Full: %s" % (motif, seq)

        if motif[5] == "Y" and len(y_seqs) < pYn:
            y_seqs.append(Seq(motif, IUPAC.protein))

        if motif[5] == "S" and len(s_seqs) < pSn:
            s_seqs.append(Seq(motif, IUPAC.protein))

        if motif[5] == "T" and len(t_seqs) < pTn:
            t_seqs.append(Seq(motif, IUPAC.protein))

    return y_seqs + s_seqs + t_seqs


###------------ Other EM-housekeeping functions  ------------------###

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


def GmmpCompatibleWithSeqScores(gmm_pred, distance_method):
    """ Make data and sequencec scores as close in magnitude as possible. """
    if distance_method == "PAM250":
        gmmp = gmm_pred * 100
    elif distance_method == "Binomial":
        gmm_pred[gmm_pred == 1] = 0.9999999999999
        gmmp = np.log(1 - gmm_pred)
    else:
        print("Distance method not regonized")
        raise SystemExit
    return gmmp


def preprocess_seqs(X, pYTS):
    """ Filter out any sequences with different than the specified central p-residue
    and/or any containing gaps."""
    X = X[~X["Sequence"].str.contains("-")]

    Xidx = []
    for seq in X["Sequence"]:
        Xidx.append(seq[5] == pYTS.lower())
    return X.iloc[Xidx, :]


def ForegroundSeqs(sequences):
    """ Build Background data set for either "Y", "S", or "T". """
    seqs = []
    yts = ["Y", "T", "S"]
    for motif in sequences:
        motif = motif.upper()
        assert "-" not in motif, "gap in motif"
        assert motif[5] in yts, "WRONG CENTRAL AMINO ACID" 
        seqs.append(Seq(motif, IUPAC.protein))
    return seqs
