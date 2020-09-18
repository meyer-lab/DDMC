import numpy as np
import scipy.stats as sp
import scipy.special as sc
from Bio.Align import substitution_matrices
from numba import njit, prange


class PAM250():
    def __init__(self, info, background, SeqWeight):
        self.d = 1
        self.name = "PAM250"
        self.SeqWeight = SeqWeight

        if isinstance(background, bool):
            seqs = [s.upper() for s in info["Sequence"]]
            # Compute all pairwise distances and generate seq vs seq to score dictionary
            self.background = MotifPam250Scores(seqs)
        else:
            self.background = background

        self.background = np.array(self.background, dtype=np.float)
        self.weights = sp.beta.rvs(a=10, b=10, size=len(info["Sequence"]))
        self.logWeights = np.log(self.weights)
        self.from_summaries()

    def summarize(self, _, w):
        self.weights = w

    def log_probability(self, X):
        return self.SeqWeight * self.logWeights[int(np.squeeze(X))]

    def from_summaries(self, inertia=0.0):
        """ Update the underlying distribution. No inertia used. """
        if np.sum(self.weights) == 0.0:
            self.logWeights = np.average(self.background, axis=0)
        else:
            self.logWeights = np.average(self.background, weights=self.weights, axis=0)

    def clear_summaries(self):
        """ Clear the summary statistics stored in the object. Not needed here. """
        return


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
