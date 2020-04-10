"""Sequence Analysis Functions. """

import os
import re
import math
from collections import defaultdict
from functools import lru_cache
import numpy as np
import pandas as pd
from Bio import SeqIO, motifs
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.SubsMat import MatrixInfo
from scipy.stats import binom
from pomegranate import GeneralMixtureModel, NormalDistribution

path = os.path.dirname(os.path.abspath(__file__))


###------------ Mapping to Uniprot's proteome and Extension of Phosphosite Sequences ------------------###

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
            except:
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
            except:
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


###------------ Motif Discovery inspired by Schwartz & Gygi, Nature Biotech 2005  ------------------###
# Amino acids frequencies (http://www.tiem.utk.edu/~gross/bioed/webmodules/aminoacid.htm) used for pseudocounts,

AAfreq = {"A": 0.074, "R": 0.042, "N": 0.044, "D": 0.059, "C": 0.033, "Q": 0.058, "E": 0.037, "G": 0.074, "H": 0.029, "I": 0.038, "L": 0.076,
          "K": 0.072, "M": 0.018, "F": 0.04, "P": 0.05, "S": 0.081, "T": 0.062, "W": 0.013, "Y": 0.033, "V": 0.068}


def e_step(ABC, distance_method, GMMweight, gmmp, bg_pwm, cl_seqs, ncl):
    """ Expectation step of the EM algorithm. Used for predict and score in
    clustering.py """
    Allseqs = ForegroundSeqs(list(ABC["Sequence"]))
    cl_seqs = [ForegroundSeqs(cluster) for cluster in cl_seqs]
    labels, scores = [], []

    binoM = BPM(cl_seqs, distance_method, bg_pwm)
    for j, motif in enumerate(Allseqs):
        score, idx = assignSeqs(ncl, motif, distance_method, GMMweight, gmmp, j, bg_pwm, cl_seqs, binoM)
        labels.append(idx)
        scores.append(score)
    return np.array(labels), np.array(scores)


def assignSeqs(ncl, motif, distance_method, GMMweight, gmmp, j, bg_pwm, cl_seqs, binomials):
    """ Do the sequence assignment. """
    scores = []
    # Binomial Probability Matrix distance (p-values) between foreground and background sequences
    if distance_method == "Binomial":
        for z in range(ncl):
            gmm_score = gmmp.iloc[j, z] * GMMweight
            assert math.isnan(gmm_score) == False and math.isinf(gmm_score) == False, ("gmm_score is either NaN or -Inf, motif = %s" % motif)
            NumMotif = TranslateMotifsToIdx(motif, list(bg_pwm.keys()))
            BPM_score = MeanBinomProbs(binomials[z], NumMotif)
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
    """ Generate binomial probability matrix for each cluster of sequences """
    if distance_method == "Binomial":
        BPM = []
        for seqs in cl_seqs:
            freqs = frequencies(seqs)
            BPM.append(BinomialMatrix(len(seqs), freqs, bg_pwm))
    if distance_method == "PAM250":
        BPM = False
    return BPM


def TranslateMotifsToIdx(motif, aa):
    """ Convert amino acid strings into numbers. """
    ResToNum = dict(zip(aa, np.arange(len(aa))))
    NumMotif = []
    for res in list(motif):
        NumMotif.append(ResToNum[res.upper()])
    assert len(NumMotif) == len(motif)
    return NumMotif


def EM_clustering_opt(data, info, ncl, GMMweight, distance_method, max_n_iter, n_runs):
    """ Run Coclustering n times and return the best fit. """
    scores, products = [], []
    for i in range(n_runs):
        print("run: ", i)
        cl_seqs, labels, score, n_iter = EM_clustering(data, info, ncl, GMMweight, distance_method, max_n_iter)
        scores.append(score)
        products.append([cl_seqs, labels, score, n_iter])

    if distance_method == "Binomial":
        idx = np.argmin(scores)
    if distance_method == "PAM250":
        idx = np.argmax(scores)

    return products[idx][0], products[idx][1], products[idx][2], products[idx][3]


def EM_clustering(data, info, ncl, GMMweight, distance_method, max_n_iter):
    # TODO: Add assertion to make sure scores go up/down
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    ABC = pd.concat([info, data.T], axis=1)
    d = np.array(data.T)
    Allseqs = ForegroundSeqs(list(ABC["Sequence"]))

    # Initialize with gmm clusters and generate gmm pval matrix
    gmm, cl_seqs, gmmp = gmm_initialize(ABC, ncl, distance_method)

    # Background sequences
    if distance_method == "Binomial":
        bg_seqs = BackgroundSeqs(ABC)
        bg_pwm = position_weight_matrix(bg_seqs)

    if distance_method == "PAM250":
        bg_pwm = False

    # EM algorithm
    DictMotifToCluster = defaultdict(list)
    store_Clseqs, store_Dicts = [], []
    for n_iter in range(max_n_iter):
#         print("iter: ", n_iter)
        labels, scores = [], []
        seq_reassign = [[] for i in range(ncl)]
        store_Dicts.append(DictMotifToCluster)
        store_Clseqs.append(cl_seqs)
        DictMotifToCluster = defaultdict(list)

        # E step: Assignment of each peptide based on data and seq
        binoM = BPM(cl_seqs, distance_method, bg_pwm)
        for j, motif in enumerate(Allseqs):
            score, idx = assignSeqs(ncl, motif, distance_method, GMMweight, gmmp, j, bg_pwm, cl_seqs, binoM)
            labels.append(idx)
            scores.append(score)
            seq_reassign[idx].append(motif)
            DictMotifToCluster[motif].append(idx)

        # Assert there are not empty clusters before updating, otherwise re-initialize algorithm
        if False in [len(sublist) > 0 for sublist in seq_reassign]:
            print("Re-initialize GMM clusters, empty cluster(s) at iteration %s" % (n_iter))
            gmm, cl_seqs, gmmp = gmm_initialize(ABC, ncl, distance_method)
            assert cl_seqs != store_Clseqs[-1], "Same cluster assignments after re-initialization"
            assert [len(sublist) > 0 for sublist in cl_seqs], "Empty cluster(s) after re-initialization"
            continue

        # M step: Update motifs, cluster centers, and gmm probabilities
        cl_seqs = seq_reassign
        gmm.fit(d)
        gmmp = pd.DataFrame(gmm.predict_proba(d))
        gmmp = GmmpCompatibleWithSeqScores(gmmp, distance_method)

        assert isinstance(cl_seqs[0][0], Seq), ("cl_seqs not Bio.Seq.Seq, check: %s" % cl_seqs)

        if DictMotifToCluster == store_Dicts[-1]:
            if GMMweight == 0:
                assert False not in [len(set(sublist)) == 1 for sublist in list(DictMotifToCluster.values())]
            cl_seqs = [[str(seq) for seq in cluster] for cluster in cl_seqs]
            return cl_seqs, np.array(labels), np.mean(scores), n_iter

#         print(np.mean(scores))
    print("convergence has not been reached. Clusters: %s GMMweight: %s" % (ncl, GMMweight))
    cl_seqs = [[str(seq) for seq in cluster] for cluster in cl_seqs]
    return cl_seqs, np.array(labels), np.mean(scores), n_iter


@lru_cache(maxsize=9000000)
def pairwise_score(seq1: str, seq2: str) -> float:
    """ Compute distance between two kinase motifs. Note this does not account for gaps. """
    score = 0.0
    for i, s1 in enumerate(seq1):
        if i == 5:
            continue
        if (seq1[i], seq2[i]) in MatrixInfo.pam250:
            score += MatrixInfo.pam250[(s1, seq2[i])]
        else:
            score += MatrixInfo.pam250[(seq2[i], s1)]

    return score


def GmmpCompatibleWithSeqScores(gmm_pred, distance_method):
    """ Make data and sequencec scores as close in magnitude as possible. """
    if distance_method == "PAM250":
        gmmp = gmm_pred * 100
    if distance_method == "Binomial":
        gmmp = pd.DataFrame(np.log(1 - gmm_pred.replace({float(1): float(0.9999999999999)})))
    return gmmp


def gmm_initialize(X, ncl, distance_method):
    """ Return peptides data set including its labels and pvalues matrix. """
    d = X.select_dtypes(include=['float64'])
    labels = [0, 0, 0]
    while len(set(labels)) < ncl:
        gmm = GeneralMixtureModel.from_samples(NormalDistribution, X=d, n_components=ncl, max_iterations=1)
        labels = gmm.predict(d)

    gmm_pred = pd.DataFrame(gmm.predict_proba(d))
    gmmp = GmmpCompatibleWithSeqScores(gmm_pred, distance_method)

    X["GMM_cluster"] = labels
    init_clusters = [ForegroundSeqs(list(X[X["GMM_cluster"] == i]["Sequence"])) for i in range(ncl)]
    return gmm, init_clusters, gmmp


def preprocess_seqs(X, pYTS):
    """ Filter out any sequences with different than the specified central p-residue
    and/or any containing gaps."""
    X = X[~X["Sequence"].str.contains("-")]

    Xidx = []
    for seq in X["Sequence"]:
        Xidx.append(seq[5] == pYTS.lower())
    return X.iloc[Xidx, :]


def BackgroundSeqs(X):
    """ Build Background data set with the same proportion of pY, pT, and pS motifs as in the foreground set of sequences.
    Note this PsP data set contains 51976 pY, 226131 pS, 81321 pT
    Source: https://www.phosphosite.org/staticDownloads.action -
    Phosphorylation_site_dataset.gz - Last mod: Wed Dec 04 14:56:35 EST 2019
    Cite: Hornbeck PV, Zhang B, Murray B, Kornhauser JM, Latham V, Skrzypek E PhosphoSitePlus, 2014: mutations,
    PTMs and recalibrations. Nucleic Acids Res. 2015 43:D512-20. PMID: 25514926 """
    #Get porportion of psite types in foreground set
    forseqs = list(X["Sequence"])
    forw_pYn, forw_pSn, forw_pTn, _ = CountPsiteTypes(forseqs, 5)
    forw_tot = forw_pYn + forw_pSn + forw_pTn

    pYf = forw_pYn / forw_tot
    pSf = forw_pSn / forw_tot
    pTf = forw_pTn / forw_tot

    #Import backgroun sequences file
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
        assert len(motif) == 11, "Wrong sequence length. Sliced: %s, Full: %s" % (motif, seq)
        assert motif[5].lower() in pR, "Wrong central AA in background set. Sliced: %s, Full: %s" % (motif, seq)

        if motif[5] == "Y" and len(y_seqs) < pYn:
            y_seqs.append(Seq(motif, IUPAC.protein))

        if motif[5] == "S" and len(s_seqs) < pSn:
            s_seqs.append(Seq(motif, IUPAC.protein))

        if motif[5] == "T" and len(t_seqs) < pTn:
            t_seqs.append(Seq(motif, IUPAC.protein))

    return y_seqs + s_seqs + t_seqs


def ForegroundSeqs(Allseqs):
    """ Build Background data set for either "Y", "S", or "T". """
    seqs = []
    for motif in Allseqs:
        motif = motif.upper()
        assert "-" not in motif, "gap in motif"
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


def MeanBinomProbs(BPM, motif):
    """ Take the mean of all pvalues corresponding to each motif residue. """
    probs = 0.0
    for i, aa in enumerate(motif):
        if i == 5: #Skip central AA
            continue
        probs += BPM[aa, i]
    return probs / (len(motif) - 1)
