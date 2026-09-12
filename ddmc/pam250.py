"""PAM250 sequence-distance model used by `ddmc.clustering.DDMC`.

Contains the `PAM250` class, which scores peptide sequences against each
cluster by their average PAM250 substitution-matrix similarity to the other
sequences currently assigned to that cluster, and `get_pam250_scores`, which
precomputes the full pairwise PAM250 similarity matrix used to do so.
"""

import numpy as np
from Bio.Align import substitution_matrices


class PAM250:
    """PAM250 sequence-distance model, used by `ddmc.clustering.DDMC` when
    `distance_method="PAM250"`.

    Scores each peptide sequence against a cluster by its (responsibility
    weighted) average pairwise PAM250 substitution score against every
    other sequence, using the fixed set of pairwise scores computed once at
    construction time.

    Attributes:
        background: Pairwise PAM250 similarity matrix between all input
            sequences, of shape (n_seqs, n_seqs).
        logWeights: Log-probability (average PAM250 score) of each sequence
            under each cluster's current model, of shape
            (n_seqs, n_clusters). Set to the scalar `0.0` until
            `from_summaries` is first called.
    """

    def __init__(self, seqs: list[str]):
        """
        Args:
            seqs: The length-11 peptide sequences being clustered.
        """
        # Compute all pairwise distances. Cast to float32 once here rather
        # than in from_summaries, which runs every EM iteration and would
        # otherwise re-convert this (potentially large) int8 matrix each time.
        self.background = get_pam250_scores(seqs).astype(np.float32)
        self.logWeights = 0.0

    def from_summaries(self, weightsIn: np.ndarray) -> None:
        """Update `self.logWeights` with each sequence's responsibility
        weighted average PAM250 similarity to all sequences, per cluster.

        Args:
            weightsIn: Soft cluster assignments (responsibilities) of shape
                (n_seqs, n_clusters), i.e. `exp(log_resp)` from the EM E step.
        """
        sums = np.sum(weightsIn, axis=0)
        sums = np.clip(sums, 0.00001, np.inf)  # Avoid empty cluster divide by 0
        self.logWeights = (self.background @ weightsIn) / sums


def get_pam250_scores(seqs: list[str]) -> np.ndarray:
    """Compute the full pairwise PAM250 similarity matrix between sequences.

    Args:
        seqs: Sequences (all the same length) to score pairwise.

    Returns:
        Symmetric array of shape (len(seqs), len(seqs)), where entry
        `[i, j]` is the summed PAM250 substitution score between
        `seqs[i]` and `seqs[j]` (aligned position-by-position, no gaps).
    """
    pam250 = substitution_matrices.load("PAM250")
    seq_idx = np.array(
        [[pam250.alphabet.find(aa) for aa in seq] for seq in seqs],
        dtype=np.int8,
    )

    # convert to np array
    pam250m = np.array(pam250.values(), dtype=np.int8).reshape(pam250.shape)

    # int32, not int8: an int8 accumulator overflows for realistic peptide
    # lengths (e.g. 11 tryptophans score 11 * 17 = 187, above int8's 127
    # ceiling), silently wrapping self-similarity scores to negative values.
    out = np.zeros((seq_idx.shape[0], seq_idx.shape[0]), dtype=np.int32)
    i_idx, j_idx = np.tril_indices(seq_idx.shape[0])
    out[i_idx, j_idx] = np.sum(pam250m[seq_idx[i_idx], seq_idx[j_idx]], axis=1)

    i_upper = np.triu_indices_from(out, k=1)
    out[i_upper] = out.T[i_upper]  # pylint: disable=unsubscriptable-object
    return out
