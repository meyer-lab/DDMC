"""PAM250 matrix to compute sequence distance between sequences and clusters."""

import glob
import numpy as np
import pandas as pd
import scipy.stats as sp
import scipy.special as sc
from Bio.Align import substitution_matrices
from numba import njit, prange
from pomegranate.distributions import CustomDistribution
from .binomial import AAlist


class PAM250(CustomDistribution):
    def __init__(self, seqs, SeqWeight, background=None):
        self.background = background

        if background is None:
            # Compute all pairwise distances and generate seq vs seq to score dictionary
            self.background = MotifPam250Scores(seqs)

        super().__init__(self.background.shape[0])
        self.seqs = seqs
        self.name = "PAM250"
        self.SeqWeight = SeqWeight
        self.from_summaries()

    def __reduce__(self):
        """ Serialize the distribution for pickle. """
        return unpackPAM, (self.seqs, self.SeqWeight, self.logWeights, self.frozen)

    def copy(self):
        return PAM250(self.seqs, self.SeqWeight, self.background)

    def from_summaries(self, inertia=0.0):
        """ Update the underlying distribution. No inertia used. """
        if np.sum(self.weightsIn) == 0.0:
            self.logWeights[:] = self.SeqWeight * np.average(self.background, axis=0)
        else:
            self.logWeights[:] = self.SeqWeight * np.average(self.background, weights=self.weightsIn, axis=0)

        self.logWeights[:] = self.logWeights - np.mean(self.logWeights)


class fixedMotif(CustomDistribution):
    def __init__(self, seqs, motif, SeqWeight):
        # Compute all pairwise log-likelihood of each peptide for a motif per cluster
        self.background = motifLL(seqs, motif)

        super().__init__(self.background.shape[0])
        self.seqs = seqs
        self.motif = motif
        self.name = "fixedMotif"
        self.SeqWeight = SeqWeight
        self.from_summaries()

    def __reduce__(self):
        """Serialize the distribution for pickle."""
        return unpackPAM, (self.seqs, self.motif, self.SeqWeight, self.logWeights, self.frozen)

    def copy(self):
        return fixedMotif(self.seqs, self.motif, self.SeqWeight)

    def from_summaries(self, inertia=0.0):
        """ Update the underlying distribution. No inertia used. """
        self.logWeights[:] = self.SeqWeight * self.background
        self.logWeights[:] = self.logWeights - np.mean(self.logWeights)


def motifLL(seqs, motif):
    """ Take a peptide list and one PSPL per cluster, then return the log-likelihood for each of 
    the peptides in the list for each of the PSPLs. """
    pam250 = substitution_matrices.load("PAM250")
    seqs = np.array([[pam250.alphabet.find(aa) for aa in seq] for seq in seqs], dtype=np.intp)
    seqs = np.delete(seqs, [5, 10], axis=1) # Delelte P0 and P+5 (not in PSPL motifs)
    PSPLs = PSPLdict()
    pspl = PSPLs[motif]
    motif_probs = np.zeros(seqs.shape[0])
    motif_probs[:] = np.array([np.average([pspl[seq, ii] for ii in range(len(seqs[0]))]) for seq in seqs])
    return motif_probs


def unpackPAM(seqs, sw, lw, frozen):
    """Unpack from pickling."""
    clss = PAM250(seqs, sw)
    clss.frozen = frozen
    clss.weightsIn[:] = np.exp(lw)
    clss.logWeights[:] = lw
    return clss


def MotifPam250Scores(seqs):
    """ Calculate and store all pairwise pam250 distances before starting. """
    pam250 = substitution_matrices.load("PAM250")
    seqs = np.array([[pam250.alphabet.find(aa) for aa in seq] for seq in seqs], dtype=np.intp)

    # WARNING this type can only hold -128 to 127
    out = np.zeros((seqs.shape[0], seqs.shape[0]), dtype=np.int8)
    pam250m = np.ndarray(pam250.shape, dtype=np.int8)

    # Move to a standard Numpy array
    for ii in range(pam250m.shape[0]):
        for jj in range(pam250m.shape[1]):
            pam250m[ii, jj] = pam250[ii, jj]

    out = distanceCalc(out, seqs, pam250m)

    i_upper = np.triu_indices_from(out, k=1)
    out[i_upper] = out.T[i_upper]  # pylint: disable=unsubscriptable-object
    return out


@njit(parallel=True)
def distanceCalc(out, seqs, pam250m):
    """ Perform all the pairwise distances, with Numba JIT. """
    for ii in prange(seqs.shape[0]):  # pylint: disable=not-an-iterable
        for jj in range(ii + 1):
            for zz in range(seqs.shape[1]):
                out[ii, jj] += pam250m[seqs[ii, zz], seqs[jj, zz]]

    return out


def PSPLdict():
    """Generate dictionary with kinase name-specificity profile pairs"""
    pspl_dict = {}
    # individual files
    PSPLs = glob.glob("./msresist/data/PSPL/*.csv")
    for sp in PSPLs:
        if sp == "./msresist/data/PSPL/pssm_data.csv":
            continue
        sp_mat = pd.read_csv(sp).sort_values(by="Unnamed: 0")

        if sp_mat.shape[0] > 20:  # Remove profiling of fixed pY and pT, include only natural AA
            assert np.all(sp_mat.iloc[:-2, 0] == AAlist), "aa don't match"
            sp_mat = sp_mat.iloc[:-2, 1:].values
        else:
            assert np.all(sp_mat.iloc[:, 0] == AAlist), "aa don't match"
            sp_mat = sp_mat.iloc[:, 1:].values

        if np.all(sp_mat >= 0):
            sp_mat = np.log2(sp_mat)

        pspl_dict[sp.split("PSPL/")[1].split(".csv")[0]] = sp_mat

    # NetPhores PSPL results
    f = pd.read_csv("msresist/data/PSPL/pssm_data.csv", header=None)
    matIDX = [np.arange(16) + i for i in range(0, f.shape[0], 16)]
    for ii in matIDX:
        kin = f.iloc[ii[0], 0]
        mat = f.iloc[ii[1:], :].T
        mat.columns = np.arange(mat.shape[1])
        mat = mat.iloc[:-1, 2:12].drop(8, axis=1).astype("float64").values
        mat = np.ma.log2(mat)
        mat = mat.filled(0)
        mat[mat > 3] = 3
        mat[mat < -3] = -3
        pspl_dict[kin] = mat

    return pspl_dict