import numpy as np
import scipy.stats as sp
import scipy.special as sc
from Bio.Align import substitution_matrices
from numba import njit, prange
from pomegranate.distributions import CustomDistribution


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
        """Serialize the distribution for pickle."""
        return unpackPAM, (self.seqs, self.SeqWeight, self.logWeights, self.frozen)

    def copy(self):
        return PAM250(self.seqs, self.SeqWeight, self.background)

    def from_summaries(self, inertia=0.0):
        """ Update the underlying distribution. No inertia used. """
        if np.sum(self.weightsIn) == 0.0:
            self.logWeights[:] = self.SeqWeight * np.average(self.background, axis=0)
        else:
            self.logWeights[:] = self.SeqWeight * np.average(self.background, weights=self.weightsIn, axis=0)


def unpackPAM(seqs, sw, lw, frozen):
    """Unpack from pickling."""
    clss = PAM250(seqs, sw)
    clss.frozen = frozen
    clss.weightsIn[:] = np.exp(lw)
    clss.logWeights[:] = lw
    return clss


def MotifPam250Scores(seqs):
    """ Calculate and store all pairwise pam250 distances before starting """
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
