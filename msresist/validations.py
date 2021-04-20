"""Code to perform computational validations."""

import re
import pandas as pd
import numpy as np
import seaborn as sns
from .motifs import DictProteomeNameToSeq
from .pre_processing import MeanCenter


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


def plotSubstratesPerCluster(x, model, kinase, ax):
    """Plot normalized number of substrates of a given kinase per cluster."""
    # Refine PsP K-S data set
    ks = pd.read_csv("msresist/data/Validations/Computational/Kinase_Substrate_Dataset.csv")
    ks = ks[
        (ks["KINASE"] == kinase) &
        (ks["IN_VIVO_RXN"] == "X") &
        (ks["IN_VIVO_RXN"] == "X") &
        (ks["KIN_ORGANISM"] == "human") &
        (ks["SUB_ORGANISM"] == "human")
    ]

    # Count matching substrates per cluster and normalize by cluster size
    x["cluster"] = model.labels()
    counters = {}
    for i in range(1, max(x["cluster"]) + 1):
        counter = 0
        cl = x[x["cluster"] == i]
        put_sub = list(zip(list(cl["Gene"]), list(list(cl["Position"]))))
        psp = list(zip(list(ks["SUB_GENE"]), list(ks["SUB_MOD_RSD"])))
        for sub_pos in put_sub:
            if sub_pos in psp:
                counter += 1
        counters[i] = counter / cl.shape[0]

    # Plot
    data = pd.DataFrame()
    data["Cluster"] = counters.keys()
    data["Normalized substrate count"] = counters.values()
    sns.barplot(data=data, x="Cluster", y="Normalized substrate count", color="darkblue", ax=ax, **{"linewidth": 0.5, "edgecolor": "k"})
    ax.set_title(kinase + " Substrate Enrichment")


def plotAKTprediction_EBDTvsCPTAC(ax, model, mcf7_model):
    """Plot frobenius distance between AKT and clusters 1 and 4 from the EBDT MCF7 and CPTAC models, respectively"""
    akt_ddmc = MeanCenter(model.predict_UpstreamKinases(), mc_col=True, mc_row=True)
    akt_ddmc.columns = ["Kinase"] + list(np.arange(model.ncl) + 1)
    akt_ddmc = akt_ddmc.sort_values(by="Kinase").set_index("Kinase")[4]
    akt_ebdt = MeanCenter(mcf7_model.predict_UpstreamKinases(), mc_col=True, mc_row=True)
    akt_ebdt.columns = ["Kinase"] + list(np.arange(mcf7_model.ncl) + 1)
    akt_ebdt = akt_ebdt.sort_values(by="Kinase").set_index("Kinase")[1]
    data = pd.concat([akt_ddmc, akt_ebdt], axis=1)
    data = pd.melt(data.reset_index(), id_vars="Kinase", value_vars=list(data.columns), var_name="Cluster", value_name="Frobenius Distance")

    # Plot
    sns.stripplot(data=data, x="Cluster", y="Frobenius Distance", ax=ax)

    # Annotate
    for ii, cluster in enumerate([1, 4], start=1):
        cluster = data[data["Cluster"] == cluster]
        hits = cluster.sort_values(by="Frobenius Distance", ascending=True)
        hits.index = np.arange(hits.shape[0])
        for jj in range(1):
            ax.annotate(hits["Kinase"].iloc[jj], (ii - 1, hits["Frobenius Distance"].iloc[jj] - 0.07), fontsize=8)
    ax.legend().remove()
    ax.set_title("Kinase vs Cluster Motif")
    ax.set_xticklabels(["EBDT cluster 1", "CPTAC cluster 4"])
    ax.set_xlabel("")
