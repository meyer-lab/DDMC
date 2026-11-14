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

    def __init__(self, seqs: list[str] | np.ndarray):
        """
        Args:
            seqs: The length-11 peptide sequences being clustered.
        """
        self.seqs = np.char.upper(np.asarray(seqs, dtype=str))
        # Compute all pairwise distances. Cast to float32 once here rather
        # than in from_summaries, which runs every EM iteration and would
        # otherwise re-convert this (potentially large) int8 matrix each time.
        self.background = get_pam250_scores(self.seqs).astype(np.float32)
        self.logWeights = 0.0

    def init_ema(self, n_clusters: int) -> None:
        """Initialize EMA sufficient-statistic accumulators for mini-batch fitting."""
        self._m_ema = np.zeros((self.background.shape[0], n_clusters), dtype=np.float32)
        self._n_ema = np.zeros(n_clusters, dtype=np.float32)

    def update_ema(
        self,
        batch_idx: np.ndarray,
        weights_batch: np.ndarray,
        alpha: float,
        scale: float = 1.0,
    ) -> None:
        """Update EMA sufficient statistics from a mini-batch of responsibilities."""
        w = (weights_batch * scale).astype(np.float32)
        m_batch = self.background[:, batch_idx] @ w
        n_batch = np.sum(w, axis=0)
        self._m_ema = (1.0 - alpha) * self._m_ema + alpha * m_batch
        self._n_ema = (1.0 - alpha) * self._n_ema + alpha * n_batch

    def split_cluster(self, donor: int, target: int) -> None:
        """Halve `donor`'s EMA sufficient statistics into `target` and `donor`."""
        half_m = 0.5 * self._m_ema[:, donor]
        self._m_ema[:, target] = half_m
        self._m_ema[:, donor] = half_m
        half_n = 0.5 * self._n_ema[donor]
        self._n_ema[target] = half_n
        self._n_ema[donor] = half_n

    def finalize_ema(self) -> None:
        """No-op for PAM250 (cluster ratios are computed on demand)."""

    def estimate_log_prob_batch(self, batch_idx: np.ndarray) -> np.ndarray:
        """Evaluate sequence log-probabilities on a mini-batch of peptides."""
        sums = np.clip(self._n_ema, 1e-5, np.inf)
        return self._m_ema[batch_idx] / sums

    def compute_log_weights(self) -> None:
        """Update `self.logWeights` for all sequences from the current EMA."""
        sums = np.clip(self._n_ema, 1e-5, np.inf)
        self.logWeights = self._m_ema / sums

    def score_sequences(self, seqs: list[str] | np.ndarray) -> np.ndarray:
        """Score unseen length-11 peptide sequences against the fitted cluster
        PAM250 similarity profiles.

        Args:
            seqs: Peptide sequences to score.

        Returns:
            Average PAM250 similarity scores of shape `(len(seqs), n_clusters)`.
        """
        seqs_upper = np.char.upper(np.asarray(seqs, dtype=str))
        bg_test = get_pam250_scores(seqs_upper, self.seqs).astype(np.float32)
        return (bg_test @ self._last_weights) / self._last_sums

    def from_summaries(self, weightsIn: np.ndarray) -> None:
        """Update `self.logWeights` with each sequence's responsibility
        weighted average PAM250 similarity to all sequences, per cluster.

        Args:
            weightsIn: Soft cluster assignments (responsibilities) of shape
                (n_seqs, n_clusters), i.e. `exp(log_resp)` from the EM E step.
        """
        sums = np.sum(weightsIn, axis=0)
        sums = np.clip(sums, 0.00001, np.inf)  # Avoid empty cluster divide by 0
        self._last_weights = weightsIn.astype(np.float32, copy=False)
        self._last_sums = sums.astype(np.float32, copy=False)
        self.logWeights = (self.background @ weightsIn) / sums


def get_pam250_scores(
    seqs: list[str] | np.ndarray, seqs_b: list[str] | np.ndarray | None = None
) -> np.ndarray:
    """Compute the pairwise PAM250 similarity matrix between sequences.

    Args:
        seqs: Sequences (all the same length) to score pairwise.
        seqs_b: Optional second sequence list. If omitted, computes the
            symmetric `(len(seqs), len(seqs))` self-similarity matrix;
            otherwise computes the `(len(seqs), len(seqs_b))` cross-matrix.

    Returns:
        Array of shape `(len(seqs), len(seqs))` (or `(len(seqs), len(seqs_b))`
        when `seqs_b` is given), where entry `[i, j]` is the summed PAM250
        substitution score between `seqs[i]` and the target sequence `j`
        (aligned position-by-position, no gaps).
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
    if seqs_b is None:
        out = np.zeros((seq_idx.shape[0], seq_idx.shape[0]), dtype=np.int32)
        i_idx, j_idx = np.tril_indices(seq_idx.shape[0])
        out[i_idx, j_idx] = np.sum(pam250m[seq_idx[i_idx], seq_idx[j_idx]], axis=1)

        i_upper = np.triu_indices_from(out, k=1)
        out[i_upper] = out.T[i_upper]  # pylint: disable=unsubscriptable-object
        return out

    seq_idx_b = np.array(
        [[pam250.alphabet.find(aa) for aa in seq] for seq in seqs_b],
        dtype=np.int8,
    )
    out = np.zeros((seq_idx.shape[0], seq_idx_b.shape[0]), dtype=np.int32)
    chunk = 256
    for start in range(0, seq_idx.shape[0], chunk):
        end = min(start + chunk, seq_idx.shape[0])
        out[start:end] = np.sum(
            pam250m[seq_idx[start:end, None, :], seq_idx_b[None, :, :]],
            axis=2,
        )
    return out
