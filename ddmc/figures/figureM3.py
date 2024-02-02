"""
This creates Figure 2: Validations
"""

import numpy as np
import pandas as pd
import seaborn as sns

from ddmc.clustering import DDMC,compute_control_pssm, get_pspl_pssm_distances 
from ddmc.binomial import AAlist
from ddmc.figures.common import (
    getSetup,
    plot_motifs,
    plot_cluster_kinase_distances,
    plot_pca_on_cluster_centers,
)
from ddmc.datasets import CPTAC, EBDT
from ddmc.motifs import get_pspls


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
    p_signal = EBDT().get_p_signal()

    # Fit DDMC
    model = DDMC(
        n_components=20,
        seq_weight=5,
        distance_method="Binomial",
        random_state=10,
        max_iter=1,
    ).fit(p_signal)

    # get cluster centers
    centers = model.transform(as_df=True)

    # parse inhibitor names from sample names
    inhibitors = [s.split(".")[1].split(".")[0] for s in p_signal.columns]

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

    plot_pca_on_cluster_centers(
        centers, [ax_a, ax_b], hue_scores=is_AKTi, hue_scores_title="AKTi?"
    )

    # Plot kinase predictions for cluster 16
    plot_cluster_kinase_distances(
        model.predict_upstream_kinases()[[16]],
        model.get_pssms(PsP_background=True, clusters=[16])[0],
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

    p_signal = CPTAC().get_p_signal()

    model = DDMC(
        n_components=30,
        seq_weight=100,
        distance_method="Binomial",
        random_state=5,
        max_iter=1,
    ).fit(p_signal)

    clusters = [3, 7, 21]
    # get pssms from ddmc clusters
    pssms = model.get_pssms(PsP_background=True, clusters=clusters)

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


def plot_fig_3c():
    """Code to create hierarchical clustering of cluster 0 across treatments"""
    p_signal = EBDT().get_p_signal()
    model = DDMC(
        n_components=20,
        seq_weight=5,
        distance_method="Binomial",
        random_state=10,
        max_iter=1,
    ).fit(p_signal)
    centers = model.transform(as_df=True)
    # the labels are structured as "MCF7.<drug>.fold"
    centers.index = [i[5:-5] for i in centers.index]
    # first cluster
    center = centers.iloc[:, 0]
    lim = np.max(np.abs(center)) * 0.3
    sns.clustermap(
        center,
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