from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import numpy as np
import scipy.stats as sp
import scipy.special as sc
from Bio.Align import substitution_matrices


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
    n = len(seqs)
    pam250 = substitution_matrices.load("PAM250")
    seqs = NumSeqs(seqs, pam250.alphabet)

    # WARNING this type can only hold -128 to 127
    dtype = np.dtype(np.int8)
    shm = shared_memory.SharedMemory(create=True, size=dtype.itemsize * n * n)
    out = np.ndarray((n, n), dtype=dtype, buffer=shm.buf)

    with ProcessPoolExecutor() as e:
        for ii in range(0, n, 500):
            e.submit(innerloop, seqs, ii, 500, shm.name, out.dtype, pam250, n)

        e.shutdown()

    out = out.copy()
    shm.close()
    shm.unlink()

    i_upper = np.triu_indices(n, k=1)
    out[i_upper] = out.T[i_upper]  # pylint: disable=unsubscriptable-object
    return out


def innerloop(seqs, ii, endi, shm_name, ddtype, pam250, n: int):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    out = np.ndarray((n, n), dtype=ddtype, buffer=existing_shm.buf)

    for idxx in range(ii, ii + endi):
        for jj in range(idxx + 1):
            out[idxx, jj] = pairwise_score(seqs[idxx], seqs[jj], pam250)

    existing_shm.close()


def pairwise_score(seq1, seq2, pam250):
    """ Compute distance between two kinase motifs. Note this does not account for gaps. """
    score = 0
    for a, b in zip(seq1, seq2):
        score += pam250[a, b]
    return int(score)


def NumSeqs(seqs, alphabet):
    """Transform sequences to numeric lists to access PAM250 more efficiently."""
    return [np.array([alphabet.find(aa) for aa in seq], dtype=np.intp) for seq in seqs]
