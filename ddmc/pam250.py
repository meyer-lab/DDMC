"""PAM250 matrix to compute sequence distance between sequences and clusters."""

import numpy as np
from Bio.Align import substitution_matrices


class PAM250:
    def __init__(self, seqs):
        # Compute all pairwise distances
        self.background = MotifPam250Scores(seqs)
        self.logWeights = 0.0

    def from_summaries(self, weightsIn):
        """Update the underlying distribution."""
        sums = np.sum(weightsIn, axis=0)
        sums = np.clip(sums, 0.00001, np.inf)  # Avoid empty cluster divide by 0
        self.logWeights = (self.background @ weightsIn) / sums


def MotifPam250Scores(seqs):
    """Calculate and store all pairwise pam250 distances before starting."""
    pam250 = substitution_matrices.load("PAM250")
    seqs = np.array(
        [[pam250.alphabet.find(aa) for aa in seq] for seq in seqs], dtype=np.int8
    )

    # convert to np array
    pam250m = np.array(pam250.values(), dtype=np.int8).reshape(pam250.shape)

    out = distanceCalc(seqs, pam250m)

    i_upper = np.triu_indices_from(out, k=1)
    out[i_upper] = out.T[i_upper]  # pylint: disable=unsubscriptable-object
    return out


def distanceCalc(seqs, pam250m):
    """Calculate all the pairwise distances."""
    # WARNING this type can only hold -128 to 127
    out = np.zeros((seqs.shape[0], seqs.shape[0]), dtype=np.int8)
    
    ii_indices, jj_indices = np.triu_indices(seqs.shape[0])
    out[ii_indices, jj_indices] = np.sum(pam250m[seqs[ii_indices], seqs[jj_indices]], axis=1)
    out = out.T

    return out
