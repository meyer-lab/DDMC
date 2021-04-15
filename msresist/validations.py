"""Code to perform computational validations."""

import re
import pandas as pd
from msresist.motifs import DictProteomeNameToSeq


def preprocess_ebdt_mcf7():
    """Preprocess MCF7 mass spec data set from EBDT (Hijazi et al Nat Biotech 2020)"""
    x = pd.read_csv("msresist/data/Validations/Computational/ebdt_mcf7.csv").drop("FDR", axis=1).set_index("sh.index.sites").drop("ARPC2_HUMAN;").reset_index()
    x.insert(0, "gene", [s.split("(")[0] for s in x["sh.index.sites"]])
    x.insert(1, "pos", [re.search(r"\(([A-Za-z0-9]+)\)", s).group(1) for s in x["sh.index.sites"]])
    x = x.drop("sh.index.sites", axis=1)
    motifs, del_ids = pos_to_motif(x["gene"], x["pos"], motif_size=5)
    x = x.set_index(["gene", "pos"]).drop(del_ids).reset_index()
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


def upstreamKin_and_pdts_perCluster(model):
    """Find number of substrates per upstream kinases for each cluster determined by the putative downstream targets (PDTs)
    determined by Hijazi et al Nat Biotech 2020 (Sup Data Set 3)."""
    pdts = pd.read_csv("msresist/data/Validations/ebdt/PDTs.csv")
    ListOfUpKin = []
    for i in range(1, model.ncl + 1):
        members = pd.read_csv("msresist/data/cluster_members/ebdt_pam250_12CL_W5_members_C" + str(i) + ".csv")
        gene_pos = list(zip(members["gene"], members["pos"]))
        members = [g + "(" + p + ")" for g, p in gene_pos]
        dictKinToSubNumber = {}
        for substrate in members:
            upK = pdts[pdts["Putative Downtream Target"] == substrate]
            if upK.shape[0] == 0:
                continue
            for kin in upK["kinase"]:
                if kin not in list(dictKinToSubNumber.keys()):
                    dictKinToSubNumber[kin] = 0
                else:
                    dictKinToSubNumber[kin] += 1

        output = pd.DataFrame()
        output["upstream_kinase"] = dictKinToSubNumber.keys()
        output["num_pdts"] = dictKinToSubNumber.values()
        output = output[output["num_pdts"] != 0]
        ListOfUpKin.append(output.sort_values(by="num_pdts", ascending=False))

    return ListOfUpKin


def plotSubstratesPerCluster(x, model, kinase, ax):
    """Plot normalized number of substrates of a given kinase per cluster."""
    # Refine PsP K-S data set
    ks = pd.read_csv("msresist/data/Validations/Computational/Kinase_Substrate_Dataset.csv")
    ks = ks[
        (ks["KINASE"] == "Akt1") &
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
        put_sub = list(zip(list(cl["gene"]), list(list(cl["pos"]))))
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
