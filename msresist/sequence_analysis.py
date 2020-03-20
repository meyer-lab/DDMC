"""Sequence Analysis Functions. """

import os
import re
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from Bio import SeqIO, motifs
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.SubsMat import MatrixInfo
from functools import lru_cache
from scipy.stats import binom
from sklearn.mixture import GaussianMixture

path = os.path.dirname(os.path.abspath(__file__))


###------------ Mapping to Uniprot's proteome and Extension of Phosphosite Sequences ------------------###

def pYmotifs(ABC, ABC_names):
    " Generate pY motifs for pre-processing. "
    names, mapped_motifs, pXpos = GeneratingKinaseMotifs(ABC_names, FormatSeq(ABC))
    ABC['Protein'] = names
    ABC['Sequence'] = mapped_motifs
    return ABC.assign(Position=pXpos)


def FormatName(X):
    """ Keep only the general protein name, without any other accession information """
    longnames = [v.split("OS")[0].strip() for v in X.iloc[:, 0]]
    shortnames = [v.split("GN=")[1].split(" PE")[0].strip() for v in X.iloc[:, 0]]
    return longnames, shortnames


def FormatSeq(X):
    """ Deleting -1/-2 for mapping to uniprot's proteome"""
    return [v.split("-")[0] for v in X.iloc[:, 1]]


def DictProteomeNameToSeq(X):
    """ To generate proteom's dictionary """
    DictProtToSeq_UP = {}
    for rec2 in SeqIO.parse(X, "fasta"):
        UP_seq = str(rec2.seq)
        UP_name = rec2.description.split("HUMAN ")[1].split(" OS")[0]
        DictProtToSeq_UP[UP_name] = str(UP_seq)
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
    matchedNames = []
    for i, MS_seq in enumerate(MS_seqs):
        MS_seqU = MS_seq.upper()
        MS_name = MS_names[i].strip()
        if MS_name in ProteomeDict and MS_seqU in ProteomeDict[MS_name]:
            matchedNames.append(MS_name)
        else:
            matchedNames.append(getKeysByValue(ProteomeDict, MS_seqU)[0])
    return matchedNames


def findmotif(MS_seq, protnames, ProteomeDict, motif_size, i):
    """ For a given MS peptide, finds it in the ProteomeDict, and maps the +/-5 AA from the p-site, accounting
    for peptides phosphorylated multiple times concurrently. """
    MS_seqU = MS_seq.upper()
    MS_name = protnames[i]
    try:
        UP_seq = ProteomeDict[MS_name]
        assert MS_seqU in UP_seq, "check " + MS_name + " with seq " + MS_seq
        assert MS_name == list(ProteomeDict.keys())[list(ProteomeDict.values()).index(str(UP_seq))], \
            "check " + MS_name + " with seq " + MS_seq
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
        print("find and replace", MS_name, "in proteome_uniprot.txt. Use: ", MS_seq)
        raise

    return pos, mappedMotif


def GeneratingKinaseMotifs(names, seqs):
    """ Main function to generate motifs using 'findmotif' in parallel. """
    motif_size = 5
    proteome = open(os.path.join(path, "./data/Sequence_analysis/proteome_uniprot2019.fa"), 'r')
    ProteomeDict = DictProteomeNameToSeq(proteome)
    protnames = MatchProtNames(ProteomeDict, names, seqs)
    MS_names, mapped_motifs, uni_pos = [], [], []
    Allseqs, Testseqs = [], []

    for i, MS_seq in enumerate(seqs):
        pos, mappedMotif = findmotif(MS_seq, protnames, ProteomeDict, motif_size, i)
        Allseqs.append(MS_seq)
        MS_names.append(protnames[i])
        Testseqs.append(MS_seq)
        uni_pos.append(pos)
        mapped_motifs.append(mappedMotif)

    li_dif = [i for i in Testseqs + Allseqs if i not in Allseqs or i not in Testseqs]
    if li_dif:
        print(" Testseqs vs Allseqs may have different peptide sequences: ", li_dif)

    assert len(names) == len(MS_names), "mapping incosistent number of names" \
        + str(len(names)) + " " + str(len(MS_names))
    assert len(seqs) == len(mapped_motifs), "mapping incosistent number of peptides" \
        + str(len(seqs)) + " " + str(len(mapped_motifs))
    assert len(uni_pos) == len(seqs), "inconsistent nubmer of pX positions" \
        + str(len(seqs)) + " " + str(len(uni_pos))

    proteome.close()
    return MS_names, mapped_motifs, uni_pos


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


###------------ Motif Discovery inspired by Schwartz & Gygi, Nature Biotech 2005  ------------------###
# Amino acids frequencies (http://www.tiem.utk.edu/~gross/bioed/webmodules/aminoacid.htm) used for pseudocounts,

AAfreq = {"A": 0.074, "R": 0.042, "N": 0.044, "D": 0.059, "C": 0.033, "Q": 0.058, "E": 0.037, "G": 0.074, "H": 0.029, "I": 0.038, "L": 0.076,
          "K": 0.072, "M": 0.018, "F": 0.04, "P": 0.05, "S": 0.081, "T": 0.062, "W": 0.013, "Y": 0.033, "V": 0.068}


def e_step(ABC, distance_method, GMMweight, gmmp, bg_pwm, cl_seqs, ncl, pYTS):
    """ Expectation step of the EM algorithm. Used for predict and score in
    clustering.py """
    Allseqs = ForegroundSeqs(list(ABC.iloc[:, 1]), pYTS)
    cl_seqs = [ForegroundSeqs(cluster, pYTS) for cluster in cl_seqs]
    labels, scores = [], []

    for j, motif in enumerate(Allseqs):
        score, idx = assignSeqs(ncl, motif, distance_method, GMMweight, gmmp, j, bg_pwm, cl_seqs, pYTS)
        labels.append(idx)
        scores.append(score)
    return np.array(labels), np.array(scores)


def assignSeqs(ncl, motif, distance_method, GMMweight, gmmp, j, bg_pwm, cl_seqs, pYTS, BPM):
    """ Do the sequence assignment. """
    scores = []
    # Binomial Probability Matrix distance (p-values) between foreground and background sequences
    if distance_method == "Binomial":
        for z in range(ncl):
            gmm_score = gmmp.iloc[j, z] * GMMweight
            assert math.isnan(gmm_score) == False and math.isinf(gmm_score) == False, ("gmm_score is either NaN or -Inf, motif = %s" % motif)
            NumMotif = TranslateMotifsToIdx(motif, list(bg_pwm.keys()))
            BPM_score = MeanBinomProbs(BPM[z], NumMotif, pYTS)
            scores.append(BPM_score + gmm_score)
        score, idx = min((score, idx) for (idx, score) in enumerate(scores))

    # Average distance between each sequence and any cluster based on PAM250 substitution matrix
    if distance_method == "PAM250":
        for z in range(ncl):
            gmm_score = gmmp.iloc[j, z] / 10 * GMMweight
            PAM250_scores = [pairwise_score(str(motif), str(seq)) for seq in cl_seqs[z]]
            PAM250_score = np.mean(PAM250_scores)
            scores.append(PAM250_score + gmm_score)
        score, idx = max((score, idx) for (idx, score) in enumerate(scores))

    assert idx <= ncl - 1, ("idx out of bounds, scores list: %s" % scores)

    return score, idx


def BPM(cl_seqs, distance_method, bg_pwm):
    if distance_method == "Binomial":
        BPM = []
        for z in range(len(cl_seqs)):
            freqs = frequencies(cl_seqs[z])
            BPM.append(BinomialMatrix(len(cl_seqs[z]), freqs, bg_pwm))
    if distance_method == "PAM250":
        BPM = False
    return BPM


def TranslateMotifsToIdx(motif, aa):
    ResToNum = dict(zip(aa, np.arange(len(aa))))
    NumMotif = []
    for res in list(motif):
        NumMotif.append(ResToNum[res.upper()])
    assert len(NumMotif) == len(motif)
    return NumMotif


def EM_clustering_opt(data, info, ncl, GMMweight, distance_method, pYTS, covariance_type, max_n_iter, n=5):
    """ Run Coclustering n times and return the best fit. """
    scores, products = [], []
    for i in range(n):
        cl_seqs, labels, score, n_iter = EM_clustering(data, info, ncl, GMMweight, distance_method, pYTS, covariance_type, max_n_iter)
        scores.append(score)
        products.append([cl_seqs, labels, score, n_iter])

    if distance_method == "Binomial":
        idx = np.argmin(scores)
    if distance_method == "PAM250":
        idx = np.argmax(scores)

    return products[idx][0], products[idx][1], products[idx][2], products[idx][3]


def EM_clustering(data, info, ncl, GMMweight, distance_method, pYTS, covariance_type, max_n_iter):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    ABC = pd.concat([info, data.T], axis=1)
    d = np.array(data.T)
    Allseqs = ForegroundSeqs(list(ABC.iloc[:, 1]), pYTS)

    # Initialize with gmm clusters and generate gmm pval matrix
    gmm, cl_seqs, gmmp = gmm_initialize(ABC, ncl, covariance_type, distance_method, pYTS)

    # Background sequences
    if distance_method == "Binomial":
        bg_seqs = BackgroundSeqs(pYTS)
        bg_pwm = position_weight_matrix(bg_seqs)

    if distance_method == "PAM250":
        bg_pwm = False

    # EM algorithm
    DictMotifToCluster = defaultdict(list)
    store_Clseqs, store_Dicts = [], []
    for n_iter in range(max_n_iter):
        labels, scores = [], []
        seq_reassign = [[] for i in range(ncl)]
        store_Dicts.append(DictMotifToCluster)
        store_Clseqs.append(cl_seqs)
        DictMotifToCluster = defaultdict(list)

        # E step: Assignment of each peptide based on data and seq
        binoM = BPM(cl_seqs, distance_method, bg_pwm)
        for j, motif in enumerate(Allseqs):
            score, idx = assignSeqs(ncl, motif, distance_method, GMMweight, gmmp, j, bg_pwm, cl_seqs, pYTS, binoM)
            labels.append(idx)
            scores.append(score)
            seq_reassign[idx].append(motif)
            DictMotifToCluster[motif].append(idx)

        # Assert there are not empty clusters before updating, otherwise re-initialize algorithm
        if False in [len(sublist) > 0 for sublist in seq_reassign]:
            print("Re-initialize GMM clusters, empty cluster(s) at iteration %s" % (n_iter))
            gmm, cl_seqs, gmmp = gmm_initialize(ABC, ncl, covariance_type, distance_method, pYTS)
            assert cl_seqs != store_Clseqs[-1], "Same cluster assignments after re-initialization"
            assert False not in [len(sublist) > 0 for sublist in cl_seqs]
            continue

        # M step: Update motifs, cluster centers, and gmm probabilities
        cl_seqs = seq_reassign
        gmmp_hard = HardAssignments(labels, ncl)
        gmm._m_step(d, gmmp_hard)
        gmmp = pd.DataFrame(gmm.predict_proba(d))
        gmmp = GmmpCompatibleWithSeqScores(gmmp, distance_method)

        assert isinstance(cl_seqs[0][0], Seq), ("cl_seqs not Bio.Seq.Seq, check: %s" % cl_seqs)

        if DictMotifToCluster == store_Dicts[-1]:
            if GMMweight == 0:
                assert False not in [len(set(sublist)) == 1 for sublist in list(DictMotifToCluster.values())]
            ICs = [InformationContent(seqs) for seqs in cl_seqs]
            cl_seqs = [[str(seq) for seq in cluster] for cluster in cl_seqs]
            return cl_seqs, np.array(labels), np.mean(scores), n_iter

    print("convergence has not been reached. Clusters: %s GMMweight: %s" % (ncl, GMMweight))
    ICs = [InformationContent(seqs) for seqs in cl_seqs]
    cl_seqs = [[str(seq) for seq in cluster] for cluster in cl_seqs]
    return cl_seqs, np.array(labels), np.mean(scores), n_iter


def HardAssignments(labels, ncl):
    m = []
    for idx in labels:
        l = [0] * ncl
        l[idx] = 1
        m.append(l)
    return np.array(m)


@lru_cache(maxsize=900000)
def pairwise_score(seq1: str, seq2: str) -> float:
    """ Compute distance between two kinase motifs. Note this does not account for gaps. """
    score = 0.0
    for i in range(len(seq1)):
        if i == 5:
            continue

        if (seq1[i], seq2[i]) in MatrixInfo.pam250:
            score += MatrixInfo.pam250[(seq1[i], seq2[i])]
        else:
            score += MatrixInfo.pam250[(seq2[i], seq1[i])]

    return score


def GmmpCompatibleWithSeqScores(gmm_pred, distance_method):
    if distance_method == "PAM250":
        gmmp = gmm_pred * 100
    if distance_method == "Binomial":
        gmmp = pd.DataFrame(np.log(1 - gmm_pred.replace({float(1): float(0.9999999999999)})))
    return gmmp


def gmm_initialize(X, ncl, covariance_type, distance_method, pYTS):
    """ Return peptides data set including its labels and pvalues matrix. """
    gmm = GaussianMixture(n_components=ncl, covariance_type=covariance_type, max_iter=1).fit(X.iloc[:, 7:])
    Xcl = X.assign(GMM_cluster=gmm.predict(X.iloc[:, 7:]))
    init_clusters = [ForegroundSeqs(list(Xcl[Xcl["GMM_cluster"] == i].iloc[:, 1]), pYTS) for i in range(ncl)]
    gmm_pred = pd.DataFrame(gmm.predict_proba(X.iloc[:, 7:]))
    gmmp = GmmpCompatibleWithSeqScores(gmm_pred, distance_method)
    return gmm, init_clusters, gmmp


def preprocess_seqs(X, pYTS):
    """ Filter out any sequences with different than the specified central p-residue
    and/or any containing gaps."""
    X = X[~X.iloc[:, 1].str.contains("-")]
    Xidx = []
    for i in range(X.shape[0]):
        Xidx.append(X.iloc[i, 1][5] == pYTS.lower())
    return X.iloc[Xidx, :]


def BackgroundSeqs(pYTS):
    """ Build Background data set for either "Y", "S", or "T".
    Source: https://www.phosphosite.org/staticDownloads.action -
    Phosphorylation_site_dataset.gz - Last mod: Wed Dec 04 14:56:35 EST 2019
    Cite: Hornbeck PV, Zhang B, Murray B, Kornhauser JM, Latham V, Skrzypek E PhosphoSitePlus, 2014: mutations,
    PTMs and recalibrations. Nucleic Acids Res. 2015 43:D512-20. PMID: 25514926 """
    bg_seqs = []
    refseqs = pd.read_csv("./msresist/data/Sequence_analysis/pX_dataset_PhosphoSitePlus2019.csv").iloc[:, 1]
    for i, seq in enumerate(refseqs):
        motif = str(seq)[7 - 5:7 + 6].upper()
        if "_" in motif or "Y" != motif[5]:
            continue

        assert len(motif) == 11, "Wrong sequence length: %s" % motif

        bg_seqs.append(Seq(motif, IUPAC.protein))

    return bg_seqs


def ForegroundSeqs(Allseqs, pYTS):
    """ Build Background data set for either "Y", "S", or "T". """
    seqs = []
    for motif in Allseqs:
        motif = motif.upper()
        assert motif[5] == pYTS, "wrong central amino acid"
        assert "-" not in pYTS, "gap in motif"
        seqs.append(Seq(motif, IUPAC.protein))
    return seqs


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
    BMP[BMP == - np.inf] = np.log(float(10**(-10)))
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


def MeanBinomProbs(BPM, motif, pYTS):
    """ Take the mean of all pvalues corresponding to each motif residue. """
    probs = 0.0
    for i, aa in enumerate(motif):
        if i == 5:
            assert aa == 19, ("wrong central numeric AA (19 == Y): %s" % aa)
            continue
        probs += BPM[aa, i]
    return probs / (len(motif) - 1)
