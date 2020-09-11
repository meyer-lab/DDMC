"""Utilization of a PAM250 transition matrix to compute sequence distance between sequences and clusters."""


from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import numpy as np
from Bio.SubsMat import MatrixInfo


def assignPeptidesPAM(ncl, scores, Seq1Seq2ToScore):
    """E-step––Do the peptide assignment according to sequence and data"""
    seq_scores = np.zeros((Seq1Seq2ToScore.shape[0], ncl))

    # Average distance between each sequence and any cluster based on PAM250 substitution matrix
    for z in range(ncl):
        seq_scores[:, z] = np.average(Seq1Seq2ToScore, weights=scores[:, z], axis=0)

    return seq_scores


def MotifPam250Scores(seqs):
    """ Calculate and store all pairwise pam250 distances before starting """
    n = len(seqs)

    # WARNING this type can only hold -128 to 127
    dtype = np.dtype(np.int8)
    shm = shared_memory.SharedMemory(create=True, size=dtype.itemsize * n * n)
    out = np.ndarray((n, n), dtype=dtype, buffer=shm.buf)

    with ProcessPoolExecutor() as e:
        for ii in range(0, n, 500):
            e.submit(innerloop, seqs, ii, 500, shm.name, out.dtype, n)

        e.shutdown()

    out = out.copy()
    shm.close()
    shm.unlink()

    i_upper = np.triu_indices(n, k=1)
    out[i_upper] = out.T[i_upper]

    assert out[5, 5] == pairwise_score(seqs[5], seqs[5]), "PAM250 scores array is wrong."
    return out


def innerloop(seqs, ii, endi, shm_name, ddtype, n: int):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    out = np.ndarray((n, n), dtype=ddtype, buffer=existing_shm.buf)

    for idxx in range(ii, ii + endi):
        for jj in range(idxx + 1):
            out[idxx, jj] = pairwise_score(seqs[idxx], seqs[jj])

    existing_shm.close()


def pairwise_score(seq1: str, seq2: str) -> float:
    """ Compute distance between two kinase motifs. Note this does not account for gaps. """
    score = 0
    for a, b in zip(seq1, seq2):
        if (a, b) in MatrixInfo.pam250:
            score += MatrixInfo.pam250[(a, b)]
        else:
            score += MatrixInfo.pam250[(b, a)]
    return score
