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
from .common import (
    plot_distance_to_upstream_kinase,
    plot_motifs,
    plot_cluster_kinase_distances,
)
from ..clustering import compute_control_pssm, get_pspl_pssm_distances
from ..binomial import AAlist
from ..motifs import DictProteomeNameToSeq, get_pspls
from ..pre_processing import filter_NaNpeptides, separate_sequence_and_abundance
from sklearn.decomposition import PCA


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    axes, f = getSetup((12, 12), (3, 3), multz={3: 1})

    plot_fig_3abd(axes[0], axes[1], axes[2])

    # 3c cannot be automatically placed on figure because of seaborn limitation
    axes[3].axis("off")

    plot_fig_3e(
        axes[4],
    )

    plot_fig_3fgh(*axes[5:8])

    return f


def plot_fig_3abd(ax_a, ax_b, ax_d):
    # Import signaling data
    x = preprocess_ebdt_mcf7()
    seqs, abund = separate_sequence_and_abundance(x)

    # Fit DDMC
    ddmc = DDMC(
        seqs,
        n_components=20,
        seq_weight=5,
        distance_method="Binomial",
        random_state=10,
        max_iter=1,
    ).fit(abund)

    # get cluster centers
    centers = ddmc.transform(as_df=True)

    # parse inhibitor names from sample names
    inhibitors = [s.split(".")[1].split(".")[0] for s in abund.columns]

    # create new bool array specifying whether each inhibitor is an AKT inhibitor
    is_AKTi = [
        drug
        in [
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
        for drug in inhibitors
    ]

    # run PCA on cluster centers
    pca = PCA(n_components=2)
    scores = pca.fit_transform(centers)  # sample by PCA component
    loadings = pca.components_  # PCA component by cluster
    variance_explained = np.round(pca.explained_variance_ratio_, 2)

    # plot scores
    sns.scatterplot(
        x=scores[:, 0],
        y=scores[:, 1],
        hue=is_AKTi,
        ax=ax_a,
        **{"linewidth": 0.5, "edgecolor": "k"},
    )
    ax_a.legend(loc="lower left", prop={"size": 9}, title="AKTi", fontsize=9)
    ax_a.set_title("PCA Scores")
    ax_a.set_xlabel("PC1 (" + str(int(variance_explained[0] * 100)) + "%)", fontsize=10)
    ax_a.set_ylabel("PC2 (" + str(int(variance_explained[1] * 100)) + "%)", fontsize=10)

    # plot loadings
    sns.scatterplot(
        x=loadings[0], y=loadings[1], ax=ax_b, **{"linewidth": 0.5, "edgecolor": "k"}
    )
    ax_b.set_title("PCA Loadings")
    ax_b.set_xlabel("PC1 (" + str(int(variance_explained[0] * 100)) + "%)", fontsize=10)
    ax_b.set_ylabel("PC2 (" + str(int(variance_explained[1] * 100)) + "%)", fontsize=10)
    ax_b.legend(prop={"size": 8})
    for j, txt in enumerate(centers.columns):
        ax_b.annotate(
            txt, (loadings[0][j] + 0.001, loadings[1][j] + 0.001), fontsize=10
        )

    # Plot kinase predictions for cluster 16
    plot_cluster_kinase_distances(
        ddmc.predict_upstream_kinases()[[16]],
        ddmc.get_pssms(PsP_background=True, clusters=[16])[0],
        ax=ax_d,
    )


def plot_fig_3e(ax):
    """Plot top scoring kinase groups"""
    NPtoCumScore = {}
    X = pd.read_csv("ddmc/data/cluster_analysis/MCF7_NKIN_CL16.csv")
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
    X = X.sort_values(by="NetPhorest Score", ascending=False).iloc[:40, :]
    sns.stripplot(
        data=X,
        y="KIN Group",
        x="NetPhorest Score",
        ax=ax,
        orient="h",
        color="royalblue",
        size=5,
        **{"linewidth": 1},
        **{"edgecolor": "black"},
    )
    ax.set_title("Cluster 16—Kinase Predictions")


def plot_fig_3fgh(ax_f, ax_g, ax_h):
    # plot erk2+ pssm
    # ERK2 White lab motif
    erk2_pssm = pd.read_csv("ddmc/data/Validations/Computational/ERK2_substrates.csv")
    erk2_pssm = compute_control_pssm([s.upper() for s in erk2_pssm["Peptide"]])
    erk2_pssm = pd.DataFrame(np.clip(erk2_pssm, a_min=0, a_max=3))
    erk2_pssm.index = AAlist
    plot_motifs(erk2_pssm, ax=ax_f, titles="ERK2")

    # ERK2 prediction
    # Import signaling data
    seqs, abund = separate_sequence_and_abundance(
        filter_NaNpeptides(
            pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotifs.csv").iloc[:, 1:],
            tmt=2,
        )
    )

    # Fit DDMC
    ddmc_cptac = DDMC(
        seqs,
        n_components=30,
        seq_weight=100,
        distance_method="Binomial",
        random_state=5,
        max_iter=1,
    ).fit(abund)

    clusters = [3, 7, 21]
    # get pssms from ddmc clusters
    pssms = ddmc_cptac.get_pssms(PsP_background=True, clusters=clusters)

    # append erk2+ pssm
    pssm_names = clusters + ["ERK2+"]
    pssms = np.append(pssms, erk2_pssm.values[None, :, :], axis=0)

    # get kinase-pssm specificities
    kinases, pspls = get_pspls()
    distances = get_pspl_pssm_distances(
        pspls, pssms, as_df=True, pssm_names=pssm_names, kinases=kinases
    )

    # plot the specificities
    plot_cluster_kinase_distances(distances, pssms, ax_g)

    # plot change in specificity to ERK2 due to shuffling
    shuffled_pssms = np.array([shuffle_pssm(pssm) for pssm in pssms])
    shuffled_distances = get_pspl_pssm_distances(
        pspls, shuffled_pssms, as_df=True, pssm_names=pssm_names, kinases=kinases
    )

    # reformat data for plotting
    melt_distances = lambda ds: ds.reset_index(names="Kinase").melt(
        id_vars="Kinase", var_name="pssm_name"
    )
    distances_melt = pd.concat(
        [
            melt_distances(distances).assign(Shuffled=False),
            melt_distances(shuffled_distances).assign(Shuffled=True),
        ]
    )
    sns.stripplot(
        data=distances_melt[distances_melt["Kinase"] == "ERK2"],
        x="pssm_name",
        y="value",
        hue="Shuffled",
        ax=ax_h,
        size=8,
    )

    ax_h.set_xlabel("Cluster")
    ax_h.set_ylabel("Frobenius Distance")
    ax_h.set_title("ERK2 Shuffled Positions")
    ax_h.legend(prop={"size": 10}, loc="lower left")

    # add arrows from original to shuffled
    for i, pssm_name in enumerate(pssm_names):
        ax_h.arrow(
            i,
            distances.loc["ERK2", pssm_name] - 0.1,
            0,
            shuffled_distances.loc["ERK2", pssm_name]
            - distances.loc["ERK2", pssm_name]
            + 0.3,
            head_width=0.25,
            head_length=0.15,
            width=0.025,
            fc="black",
            ec="black",
        )


def shuffle_pssm(pssm):
    shuffled_pssm = pssm[:, np.random.permutation([0, 1, 2, 3, 4, 6, 7, 8, 9])]
    shuffled_pssm = np.insert(shuffled_pssm, 5, pssm[:, 5], axis=1)
    return np.insert(shuffled_pssm, 1, pssm[:, -1], axis=1)


def plot_fig_3c(model, cluster):
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


makeFigure()
