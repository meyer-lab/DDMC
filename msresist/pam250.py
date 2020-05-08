"""Utilization of a PAM250 transition matrix to compute sequence distance between sequences and clusters."""


from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import numpy as np
from Bio.Alphabet import IUPAC
from Bio.SubsMat import MatrixInfo


def MotifPam250Scores(seqs):
    """ Calculate and store all pairwise pam250 distances before starting """
    n = len(seqs)

    out = np.zeros((n, n), dtype=int)
    shm = shared_memory.SharedMemory(create=True, size=out.nbytes)
    out = np.ndarray(out.shape, dtype=out.dtype, buffer=shm.buf)

    with ProcessPoolExecutor(max_workers=32) as e:
        for ii in range(0, n, 500):
            e.submit(innerloop, seqs, ii, 500, shm.name, out.dtype, n)

    out = out.copy()
    shm.unlink()

    i_upper = np.triu_indices(n, k=1)
    out[i_upper] = out.T[i_upper]

    assert out[5, 5] == pairwise_score(seqs[5], seqs[5]), "PAM250 scores array is wrong."
    return out


def innerloop(seqs, ii, endi, shm_name, ddtype, n):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    out = np.ndarray((n, n), dtype=ddtype, buffer=existing_shm.buf)

    for idxx in range(ii, ii + endi):
        for jj in range(idxx + 1):
            out[idxx, jj] = pairwise_score(seqs[idxx], seqs[jj])

    existing_shm.close()


def pairwise_score(seq1: str, seq2: str) -> float:
    """ Compute distance between two kinase motifs. Note this does not account for gaps. """
    score = 0
    for i in range(len(seq1)):
        if (seq1[i], seq2[i]) in MatrixInfo.pam250:
            score += MatrixInfo.pam250[(seq1[i], seq2[i])]
        else:
            score += MatrixInfo.pam250[(seq2[i], seq1[i])]
    return score