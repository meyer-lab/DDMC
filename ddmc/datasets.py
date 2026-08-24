"""Loaders for the mass-spec datasets bundled with the package.

Contains:
    - `CPTAC`: the CPTAC lung cancer clinical phosphoproteomics cohort, plus
      accompanying clinical metadata (mutation calls, tumor/NAT status,
      hot/cold immune infiltration labels) used in the DDMC paper.
    - `EBDT`: the MCF7 kinase-inhibitor phosphoproteomics dataset from
      Hijazi et al., *Nat Biotechnol* 2020, remapped onto DDMC's length-11
      sequence-motif representation.
    - `filter_incomplete_peptides` / `select_peptide_subset`: preprocessing
      helpers for filtering a `p_signal` DataFrame by missingness or down
      to a random subset of peptides, for use before `ddmc.clustering.DDMC.fit`.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, overload

import numpy as np
import pandas as pd

from ddmc.motifs import get_proteome_name_to_seq

DATA_DIR = Path(__file__).parent / "data"


def filter_incomplete_peptides(
    p_signal: pd.DataFrame,
    sample_presence_ratio: float | None = None,
    min_experiments: int | None = None,
    sample_to_experiment: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Filters out missing values from p-signal array.

    Args:
        sample_presence_ratio: the minimum fraction of non-missing values
            allowed for a peptide before it is DROPPED.
        min_experiments: the minimum number of experiments allowed for a peptide
            before it is DROPPED. Must also pass in sample_to_experiment.
        sample_to_experiment: array of shape `len(p_signal.columns)` that maps
            each sample to an experiment (any identifier).

    Returns:
        Filtered data.
    """
    # assume that X has sequences as the index and samples as columns
    if sample_presence_ratio is not None:
        peptide_idx = (
            np.count_nonzero(~np.isnan(p_signal), axis=1) / p_signal.shape[1]
            >= sample_presence_ratio
        )
    elif min_experiments is not None:
        assert min_experiments is not None
        assert sample_to_experiment is not None
        # this is kind of confusing because of the use of numpy, but we're
        # removing rows that have less than the minimum number of experiments
        unique_experiments = np.unique(sample_to_experiment)
        experiments_grid, s_to_e_grid = np.meshgrid(
            unique_experiments, sample_to_experiment, indexing="ij"
        )
        bool_matrix = experiments_grid == s_to_e_grid
        present = ~np.isnan(p_signal.values)
        peptide_idx = (present[None, :, :] & bool_matrix[:, None, :]).any(axis=2).sum(
            axis=0
        ) >= min_experiments
    else:
        raise ValueError(
            "Must specify either a sample presence or n_experiments threshold"
        )
    return p_signal.iloc[peptide_idx]


def select_peptide_subset(
    p_signal: pd.DataFrame,
    keep_ratio: float | None = None,
    keep_num: int | None = None,
) -> pd.DataFrame:
    """
    Selects a random subset of peptides from p_signal.

    Args:
        p_signal: Phosphorylation signal, indexed by peptide sequence.
        keep_ratio: Fraction of peptides to keep; if given, overrides
            `keep_num` with `int(p_signal.shape[0] * keep_ratio)`.
        keep_num: Number of peptides to keep. Required if `keep_ratio` is
            not given.

    Returns:
        A random subset of the rows of `p_signal` (sampled with
        replacement), of shape (keep_num, p_signal.shape[1]).
    """
    if keep_ratio is not None:
        keep_num = int(p_signal.shape[0] * keep_ratio)
    return p_signal.iloc[np.random.choice(p_signal.shape[0], keep_num)]


class CPTAC:
    """Loader for the CPTAC lung cancer clinical phosphoproteomics cohort and
    its accompanying clinical metadata.

    Sample columns throughout this dataset are patient IDs, with tumor
    samples given plain (e.g. `"C3L.00001"`) and their matched adjacent
    normal tissue (NAT) samples suffixed with `".N"` (e.g. `"C3L.00001.N"`).
    """

    data_dir = DATA_DIR / "MS" / "CPTAC"

    @overload
    def get_sample_to_experiment(self, as_df: Literal[False] = False) -> np.ndarray: ...
    @overload
    def get_sample_to_experiment(self, as_df: Literal[True]) -> pd.DataFrame: ...
    def get_sample_to_experiment(
        self, as_df: bool = False
    ) -> np.ndarray | pd.DataFrame:
        """Load the mapping from sample to the TMT experiment it was run in.

        Args:
            as_df: If True, return the raw DataFrame read from
                `IDtoExperiment.csv` instead of just the experiment column.

        Returns:
            If `as_df`, the full `IDtoExperiment.csv` DataFrame. Otherwise, an
            array of shape `(n_samples,)` giving each sample's experiment
            identifier, aligned to that CSV's row order.
        """
        sample_to_experiment = pd.read_csv(self.data_dir / "IDtoExperiment.csv")
        if as_df:
            return sample_to_experiment
        return sample_to_experiment.iloc[:, 1].values

    def get_p_signal(self, min_experiments: int = 2) -> pd.DataFrame:
        """Load the CPTAC phosphorylation signal matrix.

        Args:
            min_experiments: The minimum number of TMT experiments a
                peptide must be observed in to be kept; passed to
                `filter_incomplete_peptides`.

        Returns:
            DataFrame of phosphorylation signal, indexed by the length-11
            peptide sequence, with one column per sample.
        """
        p_signal = pd.read_csv(self.data_dir / "CPTAC-preprocessedMotifs.csv").iloc[
            :, 1:
        ]
        p_signal = p_signal.set_index("Sequence")
        p_signal = p_signal.drop(columns=["Protein", "Gene", "Position"])
        return filter_incomplete_peptides(
            p_signal,
            min_experiments=min_experiments,
            sample_to_experiment=self.get_sample_to_experiment(),
        )

    def get_patients_with_nat_and_tumor(self, samples) -> np.ndarray:
        """
        Get patients that have both NAT and tumor samples.

        Args:
            samples (Sequence[str] | numpy.ndarray): Sample identifiers to
                consider (tumor samples plain, NAT samples suffixed with
                `".N"`). Pooled internal-reference channels (containing
                `"IR"`, e.g. `"Tumor.Only.IR"`) are ignored, as they are not
                real patient samples.

        Returns:
            Sorted array of patient IDs (the tumor-sample form, without
            `".N"`) present in `samples` as both a tumor and a NAT sample.
        """
        samples = np.asarray(samples, dtype=str)
        samples = samples[np.char.find(samples, "IR") == -1]
        tumor_samples = np.sort(samples[~np.char.endswith(samples, ".N")])
        nat_samples = np.sort(samples[np.char.endswith(samples, ".N")])
        tumor_patients = tumor_samples
        nat_patients = np.char.replace(nat_samples, ".N", "")
        return np.intersect1d(tumor_patients, nat_patients)

    def get_mutations(
        self, mutation_names: Sequence[str] | None = None
    ) -> pd.DataFrame:
        """Load per-patient genetic mutation calls.

        Args:
            mutation_names: If given, restrict the result to these mutation
                columns (as named in `Patient_Mutations.csv`, e.g.
                `"EGFR.mutation.status"`). Defaults to all mutation columns.

        Returns:
            Boolean DataFrame indexed by patient ID (restricted to patients
            with both a tumor and NAT sample), with one column per
            mutation, True where that patient carries the mutation.
        """
        mutations = pd.read_csv(self.data_dir / "Patient_Mutations.csv")
        mutations = mutations.set_index("Sample.ID")
        patients = self.get_patients_with_nat_and_tumor(mutations.index.values)
        mutations = mutations.loc[patients]
        if mutation_names is not None:
            mutations = mutations[mutation_names]
        return mutations.astype(bool)

    def get_hot_cold_labels(self) -> pd.Series:
        """Load per-patient immune infiltration ("hot"/"cold" tumor) labels.

        Tumor samples labeled "NAT enriched" (ambiguous/mixed signal) are
        dropped, as are NAT samples themselves (this label only applies to
        tumor samples).

        Returns:
            Boolean Series indexed by patient ID, True for immunologically
            "hot" tumors ("Hot-tumor enriched") and False for "cold" tumors
            ("Cold-tumor enriched").
        """
        hot_cold = (
            pd.read_csv(self.data_dir / "Hot_Cold.csv")
            .dropna(axis=1)
            .sort_values(by="Sample ID")
            .set_index("Sample ID")
        )["Group"]
        hot_cold = hot_cold[~hot_cold.index.str.endswith(".N")]
        hot_cold = hot_cold[hot_cold != "NAT enriched"]
        hot_cold = hot_cold.replace("Cold-tumor enriched", 0)
        hot_cold = hot_cold.replace("Hot-tumor enriched", 1)
        hot_cold = hot_cold.dropna()
        return np.squeeze(hot_cold).astype(bool)

    def get_tumor_or_nat(self, samples: Sequence[str] | pd.Index) -> np.ndarray:
        """
        Get tumor vs NAT for each of samples. Returned array contains True if
        tumor.

        Args:
            samples: Sample identifiers (tumor samples plain, NAT samples
                suffixed with `".N"`).

        Returns:
            Boolean array of shape `(len(samples),)`, aligned to `samples`,
            True where the sample is a tumor sample (not NAT).
        """
        return ~np.array([sample.endswith(".N") for sample in samples])


# MCF7 mass spec data set from EBDT (Hijazi et al Nat Biotech 2020)
class EBDT:
    """Loader for the MCF7 kinase-inhibitor phosphoproteomics dataset from
    Hijazi et al., *Nat Biotechnol* 2020. Each sample column is the
    fold-change in phosphorylation signal for MCF7 cells treated with a
    given kinase inhibitor, relative to control.
    """

    def get_p_signal(self) -> pd.DataFrame:
        """Load the EBDT phosphorylation fold-change matrix.

        Reads the raw per-site CSV, maps each site onto the human proteome
        to build DDMC's length-11 sequence-motif representation (via
        `pos_to_motif`), and drops any site that fails to map.

        Returns:
            DataFrame of phosphorylation fold-change, indexed by the
            length-11 peptide sequence, with one column per inhibitor
            treatment.
        """
        p_signal = (
            pd.read_csv(DATA_DIR / "Validations" / "Computational" / "ebdt_mcf7.csv")
            .drop("FDR", axis=1)
            .set_index("sh.index.sites")
            .drop("ARPC2_HUMAN;")
            .reset_index()
        )
        p_signal.insert(
            0, "Gene", [s.split("(")[0] for s in p_signal["sh.index.sites"]]
        )
        positions = []
        for s in p_signal["sh.index.sites"]:
            match = re.search(r"\(([A-Za-z0-9]+)\)", s)
            assert match is not None, f"Could not parse position from {s}"
            positions.append(match.group(1))
        p_signal.insert(1, "Position", positions)
        p_signal = p_signal.drop("sh.index.sites", axis=1)
        motifs, del_ids = self.pos_to_motif(p_signal["Gene"], p_signal["Position"])
        p_signal = p_signal.set_index(["Gene", "Position"]).drop(del_ids).reset_index()
        p_signal.insert(0, "Sequence", motifs)
        p_signal = p_signal.drop(columns=["Gene", "Position"])
        p_signal = p_signal.set_index("Sequence")
        return p_signal

    def pos_to_motif(
        self, genes: Sequence[str], pos: Sequence[str]
    ) -> tuple[list[str], list[list[str]]]:
        """Map p-site sequence position to uniprot's proteome and extract motifs.

        Args:
            genes: Gene name for each phosphosite (used to look up the
                protein sequence in the UniProt proteome).
            pos: Phosphosite position for each entry, formatted as the
                phosphoacceptor residue letter followed by its 1-indexed
                position in the protein (e.g. `"S104"`).

        Returns:
            A tuple `(motifs, del_ids)`:
                motifs: The length-11 sequence motif (5 AAs flanking the
                    phosphoacceptor on each side, phosphoacceptor
                    lowercased) for each successfully mapped site.
                del_ids: `[gene, pos]` pairs that could not be mapped
                    (gene missing from the proteome, position out of range,
                    or the residue at that position isn't S/T/Y), to be
                    dropped from the corresponding `p_signal` rows.
        """
        proteome = open(DATA_DIR / "Sequence_analysis" / "proteome_uniprot2019.fa")
        motif_size = 5
        ProteomeDict = get_proteome_name_to_seq(proteome, n="gene")
        motifs = []
        del_GeneToPos = []
        for gene, p in zip(genes, pos, strict=True):
            try:
                UP_seq = ProteomeDict[gene]
            except BaseException:
                del_GeneToPos.append([gene, p])
                continue
            idx = int(p[1:]) - 1
            motif = list(UP_seq[max(0, idx - motif_size) : idx + motif_size + 1])
            if (
                len(motif) != motif_size * 2 + 1
                or p[0] != motif[motif_size]
                or p[0] not in ["S", "T", "Y"]
            ):
                del_GeneToPos.append([gene, p])
                continue
            motif[motif_size] = motif[motif_size].lower()
            motifs.append("".join(motif))
        return motifs, del_GeneToPos
