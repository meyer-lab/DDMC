"""
This creates Figure 2: Validations
"""

import re
import numpy as np
import pandas as pd
import seaborn as sns
from ..clustering import DDMC
from ..binomial import AAlist
from .common import getSetup
from ..pca import plotPCA
from .common import plotDistanceToUpstreamKinase, plotMotifs
from ..clustering import compute_control_pssm
from ..binomial import AAlist
from ..motifs import DictProteomeNameToSeq
from ..pre_processing import filter_NaNpeptides


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (3, 3), multz={3: 1})

    # Import signaling data
    x = preprocess_ebdt_mcf7()
    d = x.select_dtypes(include=[float]).T
    i = x["Sequence"]

    # Fit DDMC and find centers
    model = DDMC(
        i, n_components=20, seq_weight=5, distance_method="Binomial", random_state=10
    ).fit(d)
    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.n_components) + 1
    centers.insert(0, "Inhibitor", x.columns[3:])
    centers["Inhibitor"] = [s.split(".")[1].split(".")[0] for s in centers["Inhibitor"]]

    # PCA AKT
    AKTi = [
        "GSK690693",
        "Torin1",
        "HS173",
        "GDC0941",
        "Ku0063794",
        "AZ20",
        "MK2206",
        "AZD5363",
        "GDC0068",
        "AZD6738",
        "AT13148",
        "Edelfosine",
        "GF109203X",
        "AZD8055",
    ]
    centers["AKTi"] = [drug in AKTi for drug in centers["Inhibitor"]]
    plotPCA(ax[:2], centers, 2, ["Inhibitor", "AKTi"], "Cluster", hue_scores="AKTi")
    ax[0].legend(loc="lower left", prop={"size": 9}, title="AKTi", fontsize=9)

    # Upstream Kinases AKT EBDT
    plotDistanceToUpstreamKinase(model, [16], ax=ax[2], num_hits=1)

    # first plot heatmap of clusters
    ax[3].axis("off")

    # AKT substrates bar plot
    plot_NetPhoresScoreByKinGroup(
        "ddmc/data/cluster_analysis/MCF7_NKIN_CL16.csv",
        ax[4],
        title="Cluster 16—Kinase Predictions",
        n=40,
    )

    # # ERK2 White lab motif
    erk2 = pd.read_csv("ddmc/data/Validations/Computational/ERK2_substrates.csv")
    erk2 = compute_control_pssm([s.upper() for s in erk2["Peptide"]])
    erk2 = pd.DataFrame(np.clip(erk2, a_min=0, a_max=3))
    erk2.index = AAlist
    plotMotifs(erk2, ax=ax[5], titles="ERK2")

    # ERK2 prediction
    # Import signaling data
    X = filter_NaNpeptides(
        pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:],
        tmt=2,
    )
    d = X.select_dtypes(include=[float]).T
    i = X["Sequence"]

    # Fit DDMC
    model_cptac = DDMC(
        i, n_components=30, seq_weight=100, distance_method="Binomial", random_state=5
    ).fit(d)

    s_pssms = ShuffleClusters([3, 7, 21], model_cptac, additional=erk2)
    plotDistanceToUpstreamKinase(
        model_cptac,
        [3, 7, 21],
        additional_pssms=s_pssms + [erk2],
        add_labels=["3_S", "7_S", "21_S", "ERK2+_S", "ERK2+"],
        ax=ax[-2:],
        num_hits=1,
    )

    return f


def ShuffleClusters(shuffle, model, additional=False):
    """Returns PSSMs with shuffled positions"""
    ClustersToShuffle = np.array(shuffle)
    pssms, _ = model.pssms(PsP_background=False)
    s_pssms = []
    for s in ClustersToShuffle:
        mat = ShufflePositions(pssms[s])
        s_pssms.append(mat)

    if not isinstance(additional, bool):
        mat = ShufflePositions(additional)
        s_pssms.append(mat)

    return s_pssms


def ShufflePositions(pssm):
    """Shuffles the positions of input PSSMs"""
    pssm = np.array(pssm)
    mat = pssm[:, np.random.permutation([0, 1, 2, 3, 4, 6, 7, 8, 9])]
    mat = np.insert(mat, 5, pssm[:, 5], axis=1)
    mat = np.insert(mat, 1, pssm[:, -1], axis=1)
    mat = pd.DataFrame(mat)
    mat.index = AAlist
    return mat


def plot_NetPhoresScoreByKinGroup(PathToFile, ax, n=5, title=False, color="royalblue"):
    """Plot top scoring kinase groups"""
    NPtoCumScore = {}
    X = pd.read_csv(PathToFile)
    for ii in range(X.shape[0]):
        curr_NPgroup = X["netphorest_group"][ii]
        if curr_NPgroup == "any_group":
            continue
        elif curr_NPgroup not in NPtoCumScore.keys():
            NPtoCumScore[curr_NPgroup] = X["netphorest_score"][ii]
        else:
            NPtoCumScore[curr_NPgroup] += X["netphorest_score"][ii]
    X = pd.DataFrame.from_dict(NPtoCumScore, orient="index").reset_index()
    X.columns = ["KIN Group", "NetPhorest Score"]
    X["KIN Group"] = [s.split("_")[0] for s in X["KIN Group"]]
    X = X.sort_values(by="NetPhorest Score", ascending=False).iloc[:n, :]
    sns.stripplot(
        data=X,
        y="KIN Group",
        x="NetPhorest Score",
        ax=ax,
        orient="h",
        color=color,
        size=5,
        **{"linewidth": 1},
        **{"edgecolor": "black"},
    )
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Kinase Predictions")


def plotMCF7AKTclustermap(model, cluster):
    """Code to create hierarchical clustering of cluster 1 across treatments"""
    c1 = pd.DataFrame(model.transform()[:, cluster - 1])
    X = pd.read_csv("ddmc/data/Validations/Computational/ebdt_mcf7.csv")
    index = [col.split("7.")[1].split(".")[0] for col in X.columns[2:]]
    c1["Inhibitor"] = index
    c1 = c1.set_index("Inhibitor")
    lim = np.max(np.abs(c1)) * 0.3
    g = sns.clustermap(
        c1,
        method="centroid",
        cmap="bwr",
        robust=True,
        vmax=lim,
        vmin=-lim,
        row_cluster=True,
        col_cluster=False,
        figsize=(2, 15),
        yticklabels=True,
        xticklabels=False,
    )


def preprocess_ebdt_mcf7():
    """Preprocess MCF7 mass spec data set from EBDT (Hijazi et al Nat Biotech 2020)"""
    x = (
        pd.read_csv("ddmc/data/Validations/Computational/ebdt_mcf7.csv")
        .drop("FDR", axis=1)
        .set_index("sh.index.sites")
        .drop("ARPC2_HUMAN;")
        .reset_index()
    )
    x.insert(0, "Gene", [s.split("(")[0] for s in x["sh.index.sites"]])
    x.insert(
        1,
        "Position",
        [re.search(r"\(([A-Za-z0-9]+)\)", s).group(1) for s in x["sh.index.sites"]],
    )
    x = x.drop("sh.index.sites", axis=1)
    motifs, del_ids = pos_to_motif(x["Gene"], x["Position"], motif_size=5)
    x = x.set_index(["Gene", "Position"]).drop(del_ids).reset_index()
    x.insert(0, "Sequence", motifs)
    return x


def pos_to_motif(genes, pos, motif_size=5):
    """Map p-site sequence position to uniprot's proteome and extract motifs."""
    proteome = open("ddmc/data/Sequence_analysis/proteome_uniprot2019.fa", "r")
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
