"""PAM250 matrix to compute sequence distance between sequences and clusters."""

import numpy as np
from Bio.Align import substitution_matrices


class PAM250():
    def __init__(self, seqs):
        # Compute all pairwise distances
        self.background = MotifPam250Scores(seqs)
        self.logWeights = 0.0

    def from_summaries(self, weightsIn):
        """ Update the underlying distribution. """
        sums = np.sum(weightsIn, axis=0)
        sums = np.clip(sums, 0.00001, np.inf)  # Avoid empty cluster divide by 0
        self.logWeights = (self.background @ weightsIn) / sums


def MotifPam250Scores(seqs):
    """ Calculate and store all pairwise pam250 distances before starting. """
    pam250 = substitution_matrices.load("PAM250")
    seqs = np.array([[pam250.alphabet.find(aa) for aa in seq] for seq in seqs], dtype=np.int8)

    # Move to a standard Numpy array
    pam250m = np.ndarray(pam250.shape, dtype=np.int8)

    for ii in range(pam250m.shape[0]):
        for jj in range(pam250m.shape[1]):
            pam250m[ii, jj] = pam250[ii, jj]

    out = distanceCalc(seqs, pam250m)

    i_upper = np.triu_indices_from(out, k=1)
    out[i_upper] = out.T[i_upper]  # pylint: disable=unsubscriptable-object
    return out


def distanceCalc(seqs, pam250m):
    """ Perform all the pairwise distances. """
    # WARNING this type can only hold -128 to 127
    out = np.zeros((seqs.shape[0], seqs.shape[0]), dtype=np.int8)

    for ii in range(seqs.shape[0]):  # pylint: disable=not-an-iterable
        for jj in range(ii + 1):
            out[ii, jj] = np.sum(pam250m[seqs[ii, :], seqs[jj, :]])

    return out
