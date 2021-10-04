"""Code to perform computational validations."""

import re
import pandas as pd
from .motifs import DictProteomeNameToSeq


def preprocess_ebdt_mcf7():
    """Preprocess MCF7 mass spec data set from EBDT (Hijazi et al Nat Biotech 2020)"""
    x = pd.read_csv("msresist/data/Validations/Computational/ebdt_mcf7.csv").drop("FDR", axis=1).set_index("sh.index.sites").drop("ARPC2_HUMAN;").reset_index()
    x.insert(0, "Gene", [s.split("(")[0] for s in x["sh.index.sites"]])
    x.insert(1, "Position", [re.search(r"\(([A-Za-z0-9]+)\)", s).group(1) for s in x["sh.index.sites"]])
    x = x.drop("sh.index.sites", axis=1)
    motifs, del_ids = pos_to_motif(x["Gene"], x["Position"], motif_size=5)
    x = x.set_index(["Gene", "Position"]).drop(del_ids).reset_index()
    x.insert(0, "Sequence", motifs)
    return x


def pos_to_motif(genes, pos, motif_size=5):
    """Map p-site sequence position to uniprot's proteome and extract motifs."""
    proteome = open("msresist/data/Sequence_analysis/proteome_uniprot2019.fa", "r")
    ProteomeDict = DictProteomeNameToSeq(proteome, n="gene")
    motifs = []
    del_GeneToPos = []
    for gene, pos in list(zip(genes, pos)):
        try:
            UP_seq = ProteomeDict[gene]
        except BaseException:
            del_GeneToPos.append([gene, pos])
            continue
        idx = int(pos[1:]) - 1
        motif = list(UP_seq[max(0, idx - motif_size): idx + motif_size + 1])
        if len(motif) != motif_size * 2 + 1 or pos[0] != motif[motif_size] or pos[0] not in ["S", "T", "Y"]:
            del_GeneToPos.append([gene, pos])
            continue
        motif[motif_size] = motif[motif_size].lower()
        motifs.append("".join(motif))
    return motifs, del_GeneToPos
