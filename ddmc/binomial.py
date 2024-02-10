"""Binomial probability calculation to compute sequence distance between sequences and clusters."""


import numpy as np
import pandas as pd
import scipy.special as sc
from Bio import motifs
from collections import OrderedDict

# Binomial method inspired by Schwartz & Gygi's Nature Biotech 2005: doi:10.1038/nbt1146

# Amino acids frequencies (http://www.tiem.utk.edu/~gross/bioed/webmodules/aminoacid.htm) used for pseudocounts,
AAfreq: OrderedDict[str, float] = OrderedDict()
AAfreq["A"] = 0.074
AAfreq["R"] = 0.042
AAfreq["N"] = 0.044
AAfreq["D"] = 0.059
AAfreq["C"] = 0.033
AAfreq["Q"] = 0.058
AAfreq["E"] = 0.037
AAfreq["G"] = 0.074
AAfreq["H"] = 0.029
AAfreq["I"] = 0.038
AAfreq["L"] = 0.076
AAfreq["K"] = 0.072
AAfreq["M"] = 0.018
AAfreq["F"] = 0.040
AAfreq["P"] = 0.050
AAfreq["S"] = 0.081
AAfreq["T"] = 0.062
AAfreq["W"] = 0.013
AAfreq["Y"] = 0.033
AAfreq["V"] = 0.068

AAlist = list(AAfreq.keys())


def position_weight_matrix(seqs, pseudoC=AAfreq):
    """Build PWM of a given set of sequences."""
    return frequencies(seqs).normalize(pseudocounts=pseudoC)


def frequencies(seqs: list[str]):
    """Build counts matrix of a given set of sequences."""
    return motifs.create(seqs, alphabet="".join(AAlist)).counts


def GenerateBinarySeqID(seqs: list[str]) -> np.ndarray:
    """Build matrix with 0s and 1s to identify residue/position pairs for every sequence"""
    res = np.zeros((len(seqs), len(AAlist), 11), dtype=bool)
    for ii, seq in enumerate(seqs):
        for pos, aa in enumerate(seq):
            res[ii, AAlist.index(aa.upper()), pos] = 1
    return res


def BackgroundSeqs(forseqs: np.ndarray[str]) -> list[str]:
    """Build Background data set with the same proportion of pY, pT, and pS motifs as in the foreground set of sequences.
    Note this PsP data set contains 51976 pY, 226131 pS, 81321 pT
    Source: https://www.phosphosite.org/staticDownloads.action -
    Phosphorylation_site_dataset.gz - Last mod: Wed Dec 04 14:56:35 EST 2019
    Cite: Hornbeck PV, Zhang B, Murray B, Kornhauser JM, Latham V, Skrzypek E PhosphoSitePlus, 2014: mutations,
    PTMs and recalibrations. Nucleic Acids Res. 2015 43:D512-20. PMID: 25514926"""
    # Get porportion of psite types in foreground set
    forw_pYn, forw_pSn, forw_pTn = CountPsiteTypes(forseqs)
    forw_tot = forw_pYn + forw_pSn + forw_pTn

    pYf = forw_pYn / forw_tot
    pSf = forw_pSn / forw_tot
    pTf = forw_pTn / forw_tot

    # Import backgroun sequences file
    PsP = pd.read_csv(
        "./ddmc/data/Sequence_analysis/pX_dataset_PhosphoSitePlus2019.csv"
    )
    PsP = PsP[~PsP["SITE_+/-7_AA"].str.contains("_")]
    PsP = PsP[~PsP["SITE_+/-7_AA"].str.contains("X")]
    refseqs = list(PsP["SITE_+/-7_AA"])
    len_bg = int(len(refseqs))
    backg_pYn, _, _ = CountPsiteTypes(refseqs)

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


def BackgProportions(refseqs: list[str], pYn: int, pSn: int, pTn: int) -> list[str]:
    """Provided the proportions, add peptides to background set."""
    y_seqs: list[str] = []
    s_seqs: list[str] = []
    t_seqs: list[str] = []

    pR = ["y", "t", "s"]
    for seq in refseqs:
        if seq[7] not in pR:
            continue

        motif = str(seq)[7 - 5 : 7 + 6].upper()
        assert len(motif) == 11, "Wrong sequence length. Sliced: %s, Full: %s" % (
            motif,
            seq,
        )
        assert (
            motif[5].lower() in pR
        ), "Wrong central AA in background set. Sliced: %s, Full: %s" % (motif, seq)

        if motif[5] == "Y" and len(y_seqs) < pYn:
            y_seqs.append(motif)

        if motif[5] == "S" and len(s_seqs) < pSn:
            s_seqs.append(motif)

        if motif[5] == "T" and len(t_seqs) < pTn:
            t_seqs.append(motif)

    return y_seqs + s_seqs + t_seqs


class Binomial:
    """Definition of the binomial sequence distance distribution."""

    def __init__(self, seqs: np.ndarray[str]):
        # Background sequences
        background = position_weight_matrix(BackgroundSeqs(seqs))
        self.background = np.array([background[AA] for AA in AAlist])
        self.foreground: np.ndarray = GenerateBinarySeqID(seqs)

        self.logWeights = 0.0
        assert np.all(np.isfinite(self.background))
        assert np.all(np.isfinite(self.foreground))

    def from_summaries(self, weightsIn: np.ndarray):
        """Update the underlying distribution."""
        k = np.einsum("kji,kl->lji", self.foreground, weightsIn)
        betaA = np.sum(weightsIn, axis=0)[:, None, None] - k
        betaA = np.clip(betaA, 0.001, np.inf)
        probmat = sc.betainc(betaA, k + 1, 1 - self.background)
        tempp = np.einsum("ijk,ljk->il", self.foreground, probmat)
        self.logWeights = np.log(tempp)


def CountPsiteTypes(X: np.ndarray[str]) -> tuple[int, int, int]:
    """Count the number of different phosphorylation types in an MS data set.

    Args:
        X (list[str]): The list of peptide sequences.

    Returns:
        tuple[int, int, int]: The number of pY, pS, and pT sites.
    """
    X = np.char.upper(X)

    # Find the center amino acid
    cA = int((len(X[0]) - 1) / 2)

    phospho_aminos = [seq[cA] for seq in X]
    pS = phospho_aminos.count("S")
    pT = phospho_aminos.count("T")
    pY = phospho_aminos.count("Y")
    return pY, pS, pT
