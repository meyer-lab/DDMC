"""PAM250 sequence-distance model used by `ddmc.clustering.DDMC`.

Contains the `PAM250` class, which scores peptide sequences against each
cluster by their average PAM250 substitution-matrix similarity to the other
sequences currently assigned to that cluster, and `get_pam250_scores`, which
precomputes the full pairwise PAM250 similarity matrix used to do so.
"""

from functools import lru_cache

import numpy as np
from Bio.Align import substitution_matrices

from ddmc.binomial import PTM_TOKENS, normalize_ptm_sequences

_PTM_BASE = {"s": "S", "t": "T", "y": "Y", "k": "K"}


@lru_cache(maxsize=2)
def _load_pam250_alphabet_and_matrix(
    include_ptms: bool = False,
) -> tuple[str, np.ndarray]:
    """Return `(alphabet, matrix)` for standard or PTM-extended PAM250."""
    pam250 = substitution_matrices.load("PAM250")
    base_alpha = str(pam250.alphabet)
    base_mat = np.array(pam250.values(), dtype=np.int8).reshape(pam250.shape)
    if not include_ptms:
        return base_alpha, base_mat

    ext_alpha = base_alpha + "".join(PTM_TOKENS)
    n_base = len(base_alpha)
    n_ext = len(ext_alpha)
    ext_mat = np.zeros((n_ext, n_ext), dtype=np.int8)
    ext_mat[:n_base, :n_base] = base_mat

    for i, ptm in enumerate(PTM_TOKENS):
        idx = n_base + i
        base_idx = base_alpha.find(_PTM_BASE[ptm])
        for j, aa in enumerate(base_alpha):
            score = int(base_mat[base_idx, j])
            # Biochemical crosstalk adjustments:
            # phospho-residues gain mild electrostatic affinity for D/E;
            # acetyl-lysine loses positive-charge affinity for R.
            if ptm in ("s", "t", "y") and aa in ("D", "E"):
                score += 1
            elif ptm == "k" and aa == "R":
                score -= 1
            ext_mat[idx, j] = score
            ext_mat[j, idx] = score

    for i, p1 in enumerate(PTM_TOKENS):
        idx1 = n_base + i
        b1 = base_alpha.find(_PTM_BASE[p1])
        for j, p2 in enumerate(PTM_TOKENS):
            idx2 = n_base + j
            b2 = base_alpha.find(_PTM_BASE[p2])
            if p1 == p2:
                ext_mat[idx1, idx2] = int(base_mat[b1, b1]) + 3
            elif p1 in ("s", "t", "y") and p2 in ("s", "t", "y"):
                ext_mat[idx1, idx2] = int(base_mat[b1, b2]) + 2
            else:
                ext_mat[idx1, idx2] = int(base_mat[b1, b2])

    return ext_alpha, ext_mat


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

    def __init__(self, seqs: list[str] | np.ndarray, include_ptms: bool = False):
        """
        Args:
            seqs: The length-11 peptide sequences being clustered.
            include_ptms: If `True`, preserve modified flanking residue tokens
                (`s`, `t`, `y`, `k`) using the PTM-extended PAM250 matrix.
        """
        self.include_ptms = include_ptms
        self.seqs = normalize_ptm_sequences(seqs, include_ptms=include_ptms)
        # Compute all pairwise distances. Cast to float32 once here rather
        # than in from_summaries, which runs every EM iteration and would
        # otherwise re-convert this (potentially large) int8 matrix each time.
        self.background = get_pam250_scores(
            self.seqs, include_ptms=include_ptms
        ).astype(np.float32)
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
        seqs_norm = normalize_ptm_sequences(seqs, include_ptms=self.include_ptms)
        bg_test = get_pam250_scores(
            seqs_norm, self.seqs, include_ptms=self.include_ptms
        ).astype(np.float32)
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
    seqs: list[str] | np.ndarray,
    seqs_b: list[str] | np.ndarray | None = None,
    include_ptms: bool = False,
) -> np.ndarray:
    """Compute the pairwise PAM250 similarity matrix between sequences.

    Args:
        seqs: Sequences (all the same length) to score pairwise.
        seqs_b: Optional second sequence list. If omitted, computes the
            symmetric `(len(seqs), len(seqs))` self-similarity matrix;
            otherwise computes the `(len(seqs), len(seqs_b))` cross-matrix.
        include_ptms: If `True`, score flanking PTM tokens (`s`, `t`, `y`, `k`)
            with the PTM-extended PAM250 substitution matrix.

    Returns:
        Array of shape `(len(seqs), len(seqs))` (or `(len(seqs), len(seqs_b))`
        when `seqs_b` is given), where entry `[i, j]` is the summed PAM250
        substitution score between `seqs[i]` and the target sequence `j`
        (aligned position-by-position, no gaps).
    """
    alphabet, pam250m = _load_pam250_alphabet_and_matrix(include_ptms)
    seq_idx = np.array(
        [[alphabet.find(aa) for aa in seq] for seq in seqs],
        dtype=np.int8,
    )

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
        [[alphabet.find(aa) for aa in seq] for seq in seqs_b],
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
