"""Binomial probability calculation to compute sequence distance between sequences and clusters."""


import numpy as np
import pandas as pd
from Bio import motifs
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from scipy.stats import binom
from msresist.motifs import CountPsiteTypes

# Binomial method inspired by Schwartz & Gygi's Nature Biotech 2005: doi:10.1038/nbt1146

# Amino acids frequencies (http://www.tiem.utk.edu/~gross/bioed/webmodules/aminoacid.htm) used for pseudocounts,
AAfreq = {
    "A": 0.074,
    "R": 0.042,
    "N": 0.044,
    "D": 0.059,
    "C": 0.033,
    "Q": 0.058,
    "E": 0.037,
    "G": 0.074,
    "H": 0.029,
    "I": 0.038,
    "L": 0.076,
    "K": 0.072,
    "M": 0.018,
    "F": 0.04,
    "P": 0.05,
    "S": 0.081,
    "T": 0.062,
    "W": 0.013,
    "Y": 0.033,
    "V": 0.068,
}


def GenerateBPM(cl_seqs, bg_pwm):
    """ Generate binomial probability matrix for each cluster of sequences """
    return [BinomialMatrix(len(seqs), frequencies(seqs), bg_pwm) for seqs in cl_seqs]


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
    assert list(k.keys()) == ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    assert list(p.keys()) == list(k.keys())
    BMP = binom.pmf(k=list(k.values()), n=n, p=list(p.values()), loc=0)
    # make the p-value of Y at pos 0 close to 0 to avoid log(0) = -inf
    BMP[BMP == -np.inf] = np.amin(np.array(BMP)[BMP != -np.inf])
    return BMP


def ExtractMotif(BMP, freqs, pvalCut=10 ** (-4), occurCut=7):
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

    return "".join(motif)


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


def BackgroundSeqs(forseqs):
    """ Build Background data set with the same proportion of pY, pT, and pS motifs as in the foreground set of sequences.
    Note this PsP data set contains 51976 pY, 226131 pS, 81321 pT
    Source: https://www.phosphosite.org/staticDownloads.action -
    Phosphorylation_site_dataset.gz - Last mod: Wed Dec 04 14:56:35 EST 2019
    Cite: Hornbeck PV, Zhang B, Murray B, Kornhauser JM, Latham V, Skrzypek E PhosphoSitePlus, 2014: mutations,
    PTMs and recalibrations. Nucleic Acids Res. 2015 43:D512-20. PMID: 25514926 """
    # Get porportion of psite types in foreground set
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

        motif = str(seq)[7 - 5: 7 + 6].upper()
        assert len(motif) == 11, "Wrong sequence length. Sliced: %s, Full: %s" % (motif, seq)
        assert motif[5].lower() in pR, "Wrong central AA in background set. Sliced: %s, Full: %s" % (motif, seq)

        if motif[5] == "Y" and len(y_seqs) < pYn:
            y_seqs.append(Seq(motif, IUPAC.protein))

        if motif[5] == "S" and len(s_seqs) < pSn:
            s_seqs.append(Seq(motif, IUPAC.protein))

        if motif[5] == "T" and len(t_seqs) < pTn:
            t_seqs.append(Seq(motif, IUPAC.protein))

    return y_seqs + s_seqs + t_seqs


def assignPeptidesBN(ncl, sequences, cl_seqs, bg_pwm, binomials, labels):
    """E-step––Do the peptide assignment according to sequence and data"""
    seq_scores = np.zeros((len(sequences), ncl))

    # Binomial Probability Matrix distance (p-values) between foreground and background sequences
    for j, motif in enumerate(sequences):
        NumMotif = TranslateMotifsToIdx(motif, list(bg_pwm.keys()))

        for z in range(ncl):
            seq_scores[j, z] = MeanBinomProbs(binomials[z], NumMotif)

    return seq_scores

