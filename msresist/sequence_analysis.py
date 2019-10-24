"""Sequence Analysis Functions. """

import os
import re
import numpy as np
import pandas as pd
from Bio import SeqIO, motifs
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.SubsMat import MatrixInfo
from scipy.stats import binom
from sklearn.mixture import GaussianMixture

path = os.path.dirname(os.path.abspath(__file__))


###------------ Mapping to Uniprot's proteome and Extension of Phosphosite Sequences ------------------###

def pYmotifs(ABC, ABC_names):
    " Generate pY motifs for pre-processing. "
    names, mapped_motifs, pXpos = GeneratingKinaseMotifs(ABC_names, FormatSeq(ABC))
    ABC['Master Protein Descriptions'] = names
    ABC['peptide-phosphosite'] = mapped_motifs
    ABC.insert(13, 'position', pXpos)
    return ABC


def FormatName(X):
    """ Keep only the general protein name, without any other accession information """
    names = []
    list(map(lambda v: names.append(v.split("OS")[0]), X.iloc[:, 1]))
    return names


def FormatSeq(X):
    """ Deleting -1/-2 for mapping to uniprot's proteome"""
    seqs = []
    list(map(lambda v: seqs.append(v.split("-")[0]), X.iloc[:, 0]))
    return seqs


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

def GeneratingKinaseMotifs(names, seqs):
    """ Generates phosphopeptide motifs accounting for doubly phospho-peptides. """
    motif_size = 5
    proteome = open(os.path.join(path, "./data/Sequence_analysis/proteome_uniprot.fa"), 'r')
    ProteomeDict = DictProteomeNameToSeq(proteome)
    protnames = MatchProtNames(ProteomeDict, names, seqs)
    MS_names, mapped_motifs, uni_pos = [], [], []
    Allseqs, Testseqs = [], []

    for i, MS_seq in enumerate(seqs):
        MS_seqU = MS_seq.upper()
        MS_name = protnames[i]
        try:
            UP_seq = ProteomeDict[MS_name]
            assert MS_seqU in UP_seq, "check " + MS_name + " with seq " + MS_seq
            assert MS_name == list(ProteomeDict.keys())[list(ProteomeDict.values()).index(str(UP_seq))], \
                "check " + MS_name + " with seq " + MS_seq
            Allseqs.append(MS_seq)
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
                uni_pos.append("Y" + str(y_idx+1) + "-p")
                MS_names.append(MS_name)
                Testseqs.append(MS_seq)
                mapped_motifs.append(makeMotif(UP_seq, MS_seq, motif_size, y_idx, center_idx, DoS_idx))

            if "y" not in MS_seq:
                pTS_idx = list(re.compile("t|s").finditer(MS_seq))
                assert len(pTS_idx) != 0
                center_idx = pTS_idx[0].start()
                ts_idx = center_idx + MatchObs[0].start()
                DoS_idx = None
                if len(pTS_idx) > 1:
                    DoS_idx = pTS_idx[1:]
                uni_pos.append(str(MS_seqU[center_idx]) + str(ts_idx+1) + "-p")
                MS_names.append(MS_name)
                Testseqs.append(MS_seq)
                mapped_motifs.append(makeMotif(UP_seq, MS_seq, motif_size, ts_idx, center_idx, DoS_idx=None))

        except BaseException:
            print("find and replace", MS_name, "in proteome_uniprot.txt. Use: ", MS_seq)
            raise


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


def makeMotif(UP_seq, MS_seq, motif_size, y_idx, center_idx, DoS_idx):
    """ Make a motif out of the matched sequences. """
    UP_seq_copy = list(UP_seq[max(0, y_idx - motif_size):y_idx + motif_size + 1])
    assert len(UP_seq_copy) > motif_size, "Size seems too small. " + UP_seq

    # If we ran off the end of the sequence at the beginning or at the end, append a gap
    if y_idx - motif_size < 0:
        for _ in range(motif_size - y_idx):
            UP_seq_copy.insert(0, "-")

    elif y_idx + motif_size + 1 > len(UP_seq):
        for _ in range(y_idx + motif_size - len(UP_seq) + 1):
            UP_seq_copy.extend("-")

    UP_seq_copy[motif_size] = UP_seq_copy[motif_size].lower()

#     Now go through and copy over phosphorylation
    if DoS_idx:
        for ppIDX in DoS_idx:
            position = ppIDX.start() - center_idx
            # If the phosphosite is within the motif
            if abs(position) < motif_size:
                editPos = position + motif_size
                UP_seq_copy[editPos] = UP_seq_copy[editPos].lower()
                assert UP_seq_copy[editPos] == MS_seq[ppIDX.start()], \
                    UP_seq_copy[editPos] + " " + MS_seq[ppIDX.start()]

    return ''.join(UP_seq_copy)



###------------ Motif Discovery inspired by Schwartz & Gygi, Nature Biotech 2005  ------------------###

""" Amino acids frequencies (http://www.tiem.utk.edu/~gross/bioed/webmodules/aminoacid.htm) used for pseudocounts,
might be able to find more reliable sources. """


AAfreq = {"A":0.074, "R":0.042, "N":0.044, "D":0.059, "C":0.033, "Q":0.058, "E":0.037, "G":0.074, "H":0.029, "I":0.038, "L":0.076, \
              "K":0.072, "M":0.018, "F":0.04, "P":0.05, "S":0.081, "T":0.062, "W":0.013, "Y":0.033, "V":0.068}


def EM_clustering(data, seqs, names, ncl, GMMweight, pYTS, distance_method, covariance_type, max_n_iter):
    """ Compute EM algorithm to cluster MS data using both data info and seq info.  """
    ABC = pd.concat([seqs, names, data.T], axis=1)
    #Initialize with gmm clusters and generate gmm pval matrix

    Cl_seqs, gmm_pvals, gmm_proba = gmm_initialCl_and_pvalues(ABC, ncl, covariance_type, pYTS)

    #Background sequences
    bg_seqs = BackgroundSeqs(pYTS)
    bg_pwm = position_weight_matrix(bg_seqs)

    #EM algorithm
    store_Cl_seqs = []
    n_iter = 0
    Allseqs = [val for sublist in Cl_seqs for val in sublist] #flatten nested clusters list
    for _ in range(max_n_iter):
        labels, store_scores = [], []
        n_iter += 1
        store_Cl_seqs.append(Cl_seqs)
        clusters = [[] for i in range(ncl)]
        for j, motif in enumerate(Allseqs):
            scores = []
            #Binomial Probability Matrix distance (p-values) between foreground and background sequences
            if distance_method == "Binomial":
                for z in range(ncl):
                    gmm_score = gmm_pvals.iloc[j, z] * GMMweight
                    freq_matrix = frequencies(Cl_seqs[z])
                    BPM = BinomialMatrix(len(Cl_seqs[z]), freq_matrix, bg_pwm)
                    BPM_score = MeanBinomProbs(BPM, motif)
                    scores.append(BPM_score + gmm_score)
                score, idx = min((score, idx) for (idx, score) in enumerate(scores))
            #Average distance between each sequence and any cluster based on PAM250 substitution matrix. 
            if distance_method == "PAM250":
                for z in range(ncl):
                    gmm_score = gmm_proba.iloc[j, z] * GMMweight
                    PAM250_scores = [pairwise_score(motif, seq, MatrixInfo.pam250)*10 for seq in Cl_seqs[z]]
                    PAM250_score = np.mean(PAM250_scores)
                    scores.append(PAM250_score + gmm_score)
                score, idx = max((score, idx) for (idx, score) in enumerate(scores))
            assert idx <= ncl - 1, ("idx out of bounds, scores list: %s" % scores)
            clusters[idx].append(motif)
            labels.append(idx)
            store_scores.append(score)

        if len(["Empty Cluster" for cluster in clusters if len(cluster)==0]) != 0:
            print("Re-initialize GMM clusters, empty cluster(s) at iteration %s" % (n_iter))
            Cl_seqs, gmm_pvals, gmm_proba = gmm_initialCl_and_pvalues(ABC, ncl, covariance_type, pYTS)
            Allseqs = [val for sublist in Cl_seqs for val in sublist]
            continue

        Cl_seqs = clusters

        #Convergence when same cluster assignments as in previous iteration
        if Cl_seqs == store_Cl_seqs[-1]:
#             print("convergence has been reached at iteration %i" % (n_iter))
            ICs = [InformationContent(seqs) for seqs in Cl_seqs]
            Cl_seqs = [[str(seq) for seq in cluster] for cluster in Cl_seqs]
            return Cl_seqs, labels, store_scores, ICs, n_iter

    print("convergence has not been reached. Clusters: %s GMMweight: %s" % (ncl, GMMweight))
    ICs = [InformationContent(seqs) for seqs in Cl_seqs]
    Cl_seqs = [[str(seq) for seq in cluster] for cluster in Cl_seqs]
    return Cl_seqs, labels, store_scores, ICs, n_iter


def match_AAs(pair, matrix):
    """ Bio.SubsMat.MatrixInfo's substitution matrices are dictionaries are triangles of the matrix.
    eg: it may include ('V', 'E') but not ('E'. 'V'). This ensures correct access to this dictionary. """
    if pair not in matrix:
        return matrix[(tuple(reversed(pair)))]
    else:
        return matrix[pair]


def pairwise_score(seq1, seq2, matrix):
    " Compute distance between two kinase motifs. Note this does not account for gaps."
    score = 0
    for i in range(len(seq1)):
        pair = (seq1[i], seq2[i])
        score += match_AAs(pair, matrix)
    return score

def gmm_initialCl_and_pvalues(X, ncl, covariance_type, pYTS):
    """ Return peptides data set including its labels and pvalues matrix. """
    gmm = GaussianMixture(n_components=ncl, covariance_type=covariance_type).fit(X.iloc[:, 3:13])
    Xcl = X.assign(GMM_cluster=gmm.predict(X.iloc[:, 3:13]))
    init_clusters = [ForegroundSeqs(list(Xcl[Xcl["GMM_cluster"] == i].iloc[:, 0]), pYTS) for i in range(ncl)]
    return init_clusters, pd.DataFrame(np.log(1 - gmm.predict_proba(X.iloc[:, 3:13]))), pd.DataFrame(gmm.predict_proba(X.iloc[:, 3:13])*100)


def preprocess_seqs(X, pYTS):
    """ Filter out any sequences with different than the specified central p-residue
    and/or any containing gaps."""
    X = X[~X.iloc[:, 0].str.contains("-")]
    Xidx = []
    for i in range(X.shape[0]):
        Xidx.append(X.iloc[i, 0][5] == pYTS.lower())
    return X.iloc[Xidx, :]


def BackgroundSeqs(pYTS):
    """ Build Background data set for either "Y", "S", or "T". """
    bg_seqs = []
    proteome = open(os.path.join(path, "./data/Sequence_analysis/proteome_uniprot.fa"), 'r')
    for prot in SeqIO.parse(proteome, "fasta"):
        seq = str(prot.seq)
        if pYTS not in seq:
            continue
        regexPattern = re.compile(pYTS)
        Y_IDXs = list(regexPattern.finditer(seq))
        for idx in Y_IDXs:
            center_idx = idx.start()
            assert seq[center_idx] == str(pYTS), "Center residue not %s" % (pYTS)
            motif = seq[center_idx-5:center_idx+6]
            if len(motif) != 11 or "X" in motif or "U" in motif:
                continue
            assert len(seq[center_idx-5:center_idx+6]) == 11, "Wrong sequence length: %s" % motif
            bg_seqs.append(Seq(motif, IUPAC.protein))

    proteome.close()
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
    return pd.DataFrame(m.counts.normalize(pseudocounts=AAfreq)).T


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
    return pd.DataFrame(m.counts).T.reset_index(drop=False)


def BinomialMatrix(n, k, p):
    """ Build binomial probability matrix. Note n is the number of sequences,
    k is the counts matrix of the MS data set, p is the pwm of the background. """
    BMP = pd.DataFrame(binom.logsf(k=k.iloc[:, 1:], n=n, p=p.iloc[:,:], loc=0))
    BMP.insert(0, "Residue", list(k.iloc[:, 0]))
    BMP.iloc[-1, 6] = np.log(float(10**(-10))) #make the p-value of Y at pos 0 close to 0 to avoid log(0) = -inf
    return BMP


def ExtractMotif(BMP, freqs, pvalCut=10**(-4), occurCut=7):
    """ Identify the most significant residue/position pairs acroos the binomial
    probability matrix meeting a probability and a occurence threshold."""
    motif = list("X"*11)
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
    BPM = BPM.set_index("Residue")
    probs = []
    for i, aa in enumerate(motif):
        if i == 5:
            continue
        probs.append(float(BPM.loc[aa][i]))
    return np.mean(probs)
