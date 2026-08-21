"""Binomial probability calculation to compute sequence distance between sequences and clusters."""

from collections import OrderedDict
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.special as sc
from Bio import motifs

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
_AAindex = {aa: i for i, aa in enumerate(AAlist)}
_pseudoCounts = np.array([AAfreq[aa] for aa in AAlist])

# Maps an ASCII byte value to its index in AAlist, for vectorized char lookup
# (see fast_position_weight_matrix). All sequences are assumed uppercase.
_AAbyteLookup = np.zeros(256, dtype=np.int64)
for _aa, _i in _AAindex.items():
    _AAbyteLookup[ord(_aa)] = _i


def position_weight_matrix(seqs, pseudoC=AAfreq):
    """Build PWM of a given set of sequences."""
    return frequencies(seqs).normalize(pseudocounts=pseudoC)


def fast_position_weight_matrix(seqs: list[str]) -> np.ndarray:
    """Build a (len(AAlist), seq_length) PWM of a given set of same-length
    sequences, equivalent to `position_weight_matrix` but without the
    overhead of Biopython's general-purpose alignment machinery."""
    seq_len = len(seqs[0])
    # Convert to fixed-width bytes and view as a 2D uint8 array so the
    # char->index lookup is a single vectorized gather instead of a nested
    # Python loop over every character of every sequence.
    seqs_bytes = np.asarray(seqs, dtype=f"S{seq_len}")
    byte_view = seqs_bytes.view(np.uint8).reshape(len(seqs), seq_len)
    idx = _AAbyteLookup[byte_view]

    counts = np.zeros((len(AAlist), seq_len))
    for pos in range(seq_len):
        counts[:, pos] = np.bincount(idx[:, pos], minlength=len(AAlist))

    return (counts + _pseudoCounts[:, None]) / (
        counts.sum(axis=0, keepdims=True) + _pseudoCounts.sum()
    )


def frequencies(seqs: list[str]):
    """Build counts matrix of a given set of sequences."""
    return motifs.create(seqs, alphabet="".join(AAlist)).counts


def GenerateBinarySeqID(seqs) -> np.ndarray:
    """Build matrix with 0s and 1s to identify residue/position pairs for every sequence"""
    res = np.zeros((len(seqs), len(AAlist), 11), dtype=bool)
    for ii, seq in enumerate(seqs):
        for pos, aa in enumerate(seq):
            res[ii, AAlist.index(aa.upper()), pos] = 1
    return res


def BackgroundSeqs(forseqs: np.ndarray) -> list[str]:
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

    refseqs, backg_pYn = _load_reference_seqs()
    len_bg = len(refseqs)

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

    # Build background sequences (cached, since for fixed reference data
    # this only depends on the pY/pS/pT proportions of the foreground set)
    return list(_cached_background_proportions(pYn, pSn, pTn))


@lru_cache(maxsize=1)
def _load_reference_seqs() -> tuple[tuple[str, ...], int]:
    """Load and filter the PhosphoSitePlus background sequences. This file
    never changes at runtime, so cache it instead of re-reading and
    re-filtering the CSV on every `BackgroundSeqs` call."""
    PsP = pd.read_csv(
        "./ddmc/data/Sequence_analysis/pX_dataset_PhosphoSitePlus2019.csv"
    )
    PsP = PsP[~PsP["SITE_+/-7_AA"].str.contains("_")]
    PsP = PsP[~PsP["SITE_+/-7_AA"].str.contains("X")]
    refseqs = tuple(PsP["SITE_+/-7_AA"])
    backg_pYn, _, _ = CountPsiteTypes(refseqs)
    return refseqs, backg_pYn


@lru_cache(maxsize=32)
def _cached_background_proportions(
    pYn: int, pSn: int, pTn: int
) -> tuple[str, ...]:
    refseqs, _ = _load_reference_seqs()
    return tuple(BackgProportions(list(refseqs), pYn, pSn, pTn))


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
        assert len(motif) == 11, f"Wrong sequence length. Sliced: {motif}, Full: {seq}"
        assert motif[5].lower() in pR, (
            f"Wrong central AA in background set. Sliced: {motif}, Full: {seq}"
        )

        if motif[5] == "Y" and len(y_seqs) < pYn:
            y_seqs.append(motif)

        if motif[5] == "S" and len(s_seqs) < pSn:
            s_seqs.append(motif)

        if motif[5] == "T" and len(t_seqs) < pTn:
            t_seqs.append(motif)

    return y_seqs + s_seqs + t_seqs


class Binomial:
    """Definition of the binomial sequence distance distribution."""

    def __init__(self, seqs: np.ndarray):
        # Background sequences
        self.background = fast_position_weight_matrix(BackgroundSeqs(seqs))
        foreground: np.ndarray = GenerateBinarySeqID(seqs)
        self.n_aa, self.n_pos = foreground.shape[1], foreground.shape[2]
        # Flattened, float view of the one-hot foreground used for fast
        # matrix multiplication in from_summaries (replacing einsum, which
        # is much slower on the boolean input and is called every EM step).
        self.foreground_flat = foreground.reshape(foreground.shape[0], -1).astype(
            np.float32
        )

        self.logWeights = 0.0
        assert np.all(np.isfinite(self.background))
        assert np.all(np.isfinite(self.foreground_flat))

    def from_summaries(self, weightsIn: np.ndarray):
        """Update the underlying distribution."""
        k_flat = weightsIn.T.astype(np.float32) @ self.foreground_flat
        k = k_flat.reshape(-1, self.n_aa, self.n_pos)
        betaA = np.sum(weightsIn, axis=0)[:, None, None] - k
        betaA = np.clip(betaA, 0.001, np.inf)
        probmat = sc.betainc(betaA, k + 1, 1 - self.background)
        probmat_flat = probmat.reshape(probmat.shape[0], -1).astype(np.float32)
        tempp = self.foreground_flat @ probmat_flat.T
        self.logWeights = np.log(tempp)


def CountPsiteTypes(X) -> tuple[int, int, int]:
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
