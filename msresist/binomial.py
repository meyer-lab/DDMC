"""Binomial probability calculation to compute sequence distance between sequences and clusters."""


import numpy as np
import pandas as pd
import scipy.stats as sp
import scipy.special as sc
from Bio import motifs
from Bio.Seq import Seq
from pomegranate.distributions import CustomDistribution

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
AAlist = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]


def position_weight_matrix(seqs, pseudoC=AAfreq):
    """Build PWM of a given set of sequences."""
    return frequencies(seqs).normalize(pseudocounts=pseudoC)


def frequencies(seqs):
    """Build counts matrix of a given set of sequences."""
    return motifs.create(seqs, alphabet=AAlist).counts


def InformationContent(seqs):
    """ The mean of the PSSM is particularly important becuase its value is equal to the
    Kullback-Leibler divergence or relative entropy, and is a measure for the information content
    of the motif compared to the background."""
    pssm = position_weight_matrix(seqs).log_odds(AAfreq)
    return pssm.mean(AAfreq)


def GenerateBinarySeqID(seqs):
    """Build matrix with 0s and 1s to identify residue/position pairs for every sequence"""
    res = np.zeros((len(seqs), len(AAlist), 11), dtype=bool)
    for ii, seq in enumerate(seqs):
        for pos, aa in enumerate(seq):
            res[ii, AAlist.index(aa.upper()), pos] = 1
    return res


def BackgroundSeqs(forseqs):
    """Build Background data set with the same proportion of pY, pT, and pS motifs as in the foreground set of sequences.
    Note this PsP data set contains 51976 pY, 226131 pS, 81321 pT
    Source: https://www.phosphosite.org/staticDownloads.action -
    Phosphorylation_site_dataset.gz - Last mod: Wed Dec 04 14:56:35 EST 2019
    Cite: Hornbeck PV, Zhang B, Murray B, Kornhauser JM, Latham V, Skrzypek E PhosphoSitePlus, 2014: mutations,
    PTMs and recalibrations. Nucleic Acids Res. 2015 43:D512-20. PMID: 25514926"""
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
    """Provided the proportions, add peptides to background set."""
    y_seqs, s_seqs, t_seqs = [], [], []
    pR = ["y", "t", "s"]
    for seq in refseqs:
        if seq[7] not in pR:
            continue

        motif = str(seq)[7 - 5: 7 + 6].upper()
        assert len(motif) == 11, "Wrong sequence length. Sliced: %s, Full: %s" % (motif, seq)
        assert motif[5].lower() in pR, "Wrong central AA in background set. Sliced: %s, Full: %s" % (motif, seq)

        if motif[5] == "Y" and len(y_seqs) < pYn:
            y_seqs.append(Seq(motif))

        if motif[5] == "S" and len(s_seqs) < pSn:
            s_seqs.append(Seq(motif))

        if motif[5] == "T" and len(t_seqs) < pTn:
            t_seqs.append(Seq(motif))

    return y_seqs + s_seqs + t_seqs


class Binomial(CustomDistribution):
    """Create a binomial distance distribution compatible with pomegranate. """

    def __init__(self, seq, seqs, SeqWeight, background=None):
        self.background = background

        if background is None:
            # Background sequences
            background = position_weight_matrix(BackgroundSeqs(seq))
            self.background = (np.array([background[AA] for AA in AAlist]), GenerateBinarySeqID(seqs))

        super().__init__(len(seqs))
        self.seq = seq
        self.seqs = seqs
        self.name = "Binomial"
        self.SeqWeight = SeqWeight
        self.from_summaries()
        assert np.all(np.isfinite(self.background[0]))
        assert np.all(np.isfinite(self.background[1]))

    def copy(self):
        return Binomial(self.seq, self.seqs, self.SeqWeight, self.background)

    def __reduce__(self):
        """Serialize the distribution for pickle."""
        return unpackBinomial, (self.seq, self.seqs, self.SeqWeight, self.logWeights, self.frozen)

    def from_summaries(self, inertia=0.0):
        """ Update the underlying distribution. No inertia used. """
        k = np.dot(self.background[1].T, self.weightsIn).T

        # The counts must be positive, so check this
        betaA = np.sum(self.weightsIn) - k
        betaA = np.clip(betaA, 0.01, np.inf)
        probmat = sc.betainc(betaA, k + 1, 1 - self.background[0])
        self.logWeights[:] = self.SeqWeight * np.log(np.tensordot(self.background[1], probmat, axes=2))


def unpackBinomial(seq, seqs, sw, lw, frozen):
    """Unpack from pickling."""
    clss = Binomial(seq, seqs, sw)
    clss.frozen = frozen
    clss.weightsIn[:] = np.exp(lw)
    clss.logWeights[:] = lw
    return clss


def CountPsiteTypes(X, cA):
    """ Count number of different phosphorylation types in a MS data set."""
    positionSeq = [seq[cA] for seq in X]
    pS = positionSeq.count("s")
    pT = positionSeq.count("t")
    pY = positionSeq.count("y")

    countt = [sum(map(str.islower, seq)) for seq in X]
    primed = sum(map(lambda i: i > 1, countt))

    return pY, pS, pT, primed
