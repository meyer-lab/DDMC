import re
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Sequence

from ddmc.motifs import get_proteome_name_to_seq

DATA_DIR = Path(__file__).parent / "data"


def filter_incomplete_peptides(
    p_signal: pd.DataFrame,
    sample_presence_ratio: float = None,
    min_experiments: int = None,
    sample_to_experiment: np.ndarray = None,
):
    # assume that X has sequences as the index and samples as columns
    if sample_presence_ratio is not None:
        peptide_idx = (
            np.count_nonzero(~np.isnan(p_signal), axis=1) / p_signal.shape[1]
            >= sample_presence_ratio
        )
    else:
        assert min_experiments is not None
        assert sample_to_experiment is not None
        unique_experiments = np.unique(sample_to_experiment)
        experiments_grid, s_to_e_grid = np.meshgrid(
            unique_experiments, sample_to_experiment, indexing="ij"
        )
        bool_matrix = experiments_grid == s_to_e_grid
        present = ~np.isnan(p_signal.values)
        peptide_idx = (present[None, :, :] & bool_matrix[:, None, :]).any(axis=2).sum(
            axis=0
        ) >= min_experiments
    return p_signal.iloc[peptide_idx]


def select_peptide_subset(
    p_signal: pd.DataFrame, keep_ratio: float = None, keep_num: int = None
):
    if keep_ratio is not None:
        keep_num = int(p_signal.shape[0] * keep_ratio)
    return p_signal.iloc[np.random.choice(p_signal.shape[0], keep_num)]


class CPTAC:
    data_dir = DATA_DIR / "MS" / "CPTAC"

    def get_sample_to_experiment(self, as_df=False):
        sample_to_experiment = pd.read_csv(self.data_dir / "IDtoExperiment.csv")
        if as_df:
            return sample_to_experiment
        return sample_to_experiment.iloc[:, 1].values

    def get_p_signal(self) -> pd.DataFrame:
        p_signal = pd.read_csv(self.data_dir / "CPTAC-preprocessedMotifs.csv").iloc[
            :, 1:
        ]
        p_signal = p_signal.set_index("Sequence")
        p_signal = p_signal.drop(columns=["Protein", "Gene", "Position"])
        return filter_incomplete_peptides(
            p_signal,
            min_experiments=2,
            sample_to_experiment=self.get_sample_to_experiment(),
        )

    def get_patients_with_nat_and_tumor(self, samples: np.ndarray[str]):
        samples = samples.astype(str)
        samples = samples[np.char.find(samples, "IR") == -1]
        tumor_samples = np.sort(samples[~np.char.endswith(samples, ".N")])
        nat_samples = np.sort(samples[np.char.endswith(samples, ".N")])
        tumor_patients = tumor_samples
        nat_patients = np.char.replace(nat_samples, ".N", "")
        return np.intersect1d(tumor_patients, nat_patients)

    def get_mutations(self, mutation_names: Sequence[str] = None):
        mutations = pd.read_csv(self.data_dir / "Patient_Mutations.csv")
        mutations = mutations.set_index("Sample.ID")
        patients = self.get_patients_with_nat_and_tumor(mutations.index.values)
        mutations = mutations.loc[patients]
        if mutation_names is not None:
            mutations = mutations[mutation_names]
        return mutations.astype(bool)

    def get_hot_cold_labels(self):
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

    def get_tumor_or_nat(self, samples: Sequence[str]) -> np.ndarray[bool]:
        return ~np.array([sample.endswith(".N") for sample in samples])


# MCF7 mass spec data set from EBDT (Hijazi et al Nat Biotech 2020)
class EBDT:
    def get_p_signal(self) -> pd.DataFrame:
        """Preprocess"""
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
        p_signal.insert(
            1,
            "Position",
            [
                re.search(r"\(([A-Za-z0-9]+)\)", s).group(1)
                for s in p_signal["sh.index.sites"]
            ],
        )
        p_signal = p_signal.drop("sh.index.sites", axis=1)
        motifs, del_ids = self.pos_to_motif(p_signal["Gene"], p_signal["Position"])
        p_signal = p_signal.set_index(["Gene", "Position"]).drop(del_ids).reset_index()
        p_signal.insert(0, "Sequence", motifs)
        p_signal = p_signal.drop(columns=["Gene", "Position"])
        p_signal = p_signal.set_index("Sequence")
        return p_signal

    def pos_to_motif(self, genes, pos):
        """Map p-site sequence position to uniprot's proteome and extract motifs."""
        proteome = open(DATA_DIR / "Sequence_analysis" / "proteome_uniprot2019.fa", "r")
        motif_size = 5
        ProteomeDict = get_proteome_name_to_seq(proteome, n="gene")
        motifs = []
        del_GeneToPos = []
        for gene, pos in list(zip(genes, pos)):
            try:
                UP_seq = ProteomeDict[gene]
            except BaseException:
                del_GeneToPos.append([gene, pos])
                continue
            idx = int(pos[1:]) - 1
            motif = list(UP_seq[max(0, idx - motif_size) : idx + motif_size + 1])
            if (
                len(motif) != motif_size * 2 + 1
                or pos[0] != motif[motif_size]
                or pos[0] not in ["S", "T", "Y"]
            ):
                del_GeneToPos.append([gene, pos])
                continue
            motif[motif_size] = motif[motif_size].lower()
            motifs.append("".join(motif))
        return motifs, del_GeneToPos
