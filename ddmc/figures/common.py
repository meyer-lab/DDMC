"""
This file contains functions that are used in multiple figures.
"""

import sys
import time
from string import ascii_uppercase
from matplotlib import gridspec, pyplot as plt, axes, rcParams
import seaborn as sns
import numpy as np
import pandas as pd
import svgutils.transform as st
import logomaker as lm
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from ..motifs import KinToPhosphotypeDict
from ddmc.binomial import AAlist
from sklearn.decomposition import PCA


rcParams["font.sans-serif"] = "Arial"


def getSetup(
    figsize: tuple[int, int],
    gridd: tuple[int, int],
    multz: None | dict = None,
    labels=True,
) -> tuple:
    """Establish figure set-up with subplots."""
    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    if multz is None:
        multz = dict()

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs1 = gridspec.GridSpec(gridd[0], gridd[1], figure=f)

    # Get list of axis objects
    x = 0
    ax = list()
    while x < gridd[0] * gridd[1]:
        if x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x : x + multz[x] + 1]))
            x += multz[x]
        x += 1

    if labels:
        subplotLabel(ax)

    return (ax, f)


def subplotLabel(axs: list[axes.Axes]):
    """Place subplot labels on the list of axes."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_uppercase[ii],
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )


def overlayCartoon(
    figFile: str, cartoonFile: str, x: float, y: float, scalee: float = 1.0
):
    """Add cartoon to a figure file."""

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale_x=scalee, scale_y=scalee)  # type: ignore

    template.append(cartoon)
    template.save(figFile)


def genFigure():
    """Main figure generation function."""
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec(f"from ddmc.figures.{nameOut} import makeFigure", globals())
    ff = makeFigure()

    if ff is not None:
        ff.savefig(
            f"./output/{nameOut}.svg", dpi=300, bbox_inches="tight", pad_inches=0
        )

    if sys.argv[1] == "M2":
        # Overlay Figure missingness cartoon
        overlayCartoon(
            f"./output/figureM2.svg",
            f"./ddmc/figures/missingness_diagram.svg",
            75,
            5,
            scalee=1.1,
        )

    if sys.argv[1] == "M5":
        # Overlay Figure tumor vs NATs heatmap
        overlayCartoon(
            f"./output/figureM5.svg",
            f"./ddmc/figures/heatmap_NATvsTumor.svg",
            50,
            0,
            scalee=0.40,
        )

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")


def plot_motifs(pssm, ax: axes.Axes, titles=False, yaxis=False):
    """Generate logo plots of a list of PSSMs"""
    pssm = pssm.T
    pssm = pd.DataFrame(pssm)
    if pssm.shape[0] == 11:
        pssm.index = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    elif pssm.shape[0] == 9:
        pssm.index = [-5, -4, -3, -2, -1, 1, 2, 3, 4]
    pssm.columns = AAlist
    logo = lm.Logo(
        pssm,
        font_name="Arial",
        vpad=0.1,
        width=0.8,
        flip_below=False,
        center_values=False,
        ax=ax,
    )
    logo.ax.set_ylabel("log_{2} (Enrichment Score)")
    logo.style_xticks(anchor=1, spacing=1)
    if titles:
        logo.ax.set_title(titles + " Motif")
    else:
        logo.ax.set_title("Motif Cluster 1")
    if yaxis:
        logo.ax.set_ylim([yaxis[0], yaxis[1]])


def plot_cluster_kinase_distances(
    distances: pd.DataFrame, pssms: np.ndarray, ax, num_hits=1
):
    pssm_names = distances.columns

    # melt distances
    distances = distances.sub(distances.mean(axis=0), axis=1)
    distances = pd.melt(
        distances.reset_index(names="Kinase"),
        id_vars="Kinase",
        value_vars=list(distances.columns),
        var_name="PSSM name",
        value_name="Frobenius Distance",
    )

    sns.stripplot(data=distances, x="PSSM name", y="Frobenius Distance", ax=ax)

    # Annotate upstream kinase predictions
    for i, pssm_name in enumerate(pssm_names):
        distances_pssm = distances[distances["PSSM name"] == pssm_name]
        distances_pssm = distances_pssm.sort_values(
            by="Frobenius Distance", ascending=True
        )
        distances_pssm = distances_pssm.reset_index(drop=True)
        # assert that the kinase phosphoacceptor and most frequent phosphoacceptor in the pssm match
        distances_pssm["Phosphoacceptor"] = [
            KinToPhosphotypeDict[kin] for kin in distances_pssm["Kinase"]
        ]
        try:
            most_frequent_phosphoacceptor = AAlist(pssms[i, 5].argmax())
        except:
            most_frequent_phosphoacceptor = "S/T"
        if most_frequent_phosphoacceptor == "S" or most_frequent_phosphoacceptor == "T":
            most_frequent_phosphoacceptor = "S/T"
        distances_pssm = distances_pssm[
            distances_pssm["Phosphoacceptor"] == most_frequent_phosphoacceptor
        ]
        for jj in range(num_hits):
            ax.annotate(
                distances_pssm["Kinase"].iloc[jj],
                (i, distances_pssm["Frobenius Distance"].iloc[jj] - 0.01),
                fontsize=8,
            )
    ax.legend().remove()
    ax.set_title("Kinase vs Cluster Motif")


def get_pvals_across_clusters(
    label: pd.Series | np.ndarray[bool], centers: pd.DataFrame | np.ndarray
) -> np.ndarray[float]:
    pvals = []
    if isinstance(centers, pd.DataFrame):
        centers = centers.values
    centers_pos = centers[label]
    centers_neg = centers[~label]
    for i in range(centers.shape[1]):
        pvals.append(mannwhitneyu(centers_pos[:, i], centers_neg[:, i])[1])
    return multipletests(pvals)[1]


def plot_p_signal_across_clusters_and_binary_feature(
    feature: pd.Series | np.ndarray[bool],
    centers: pd.DataFrame | np.ndarray,
    label_name: str,
    ax,
) -> None:
    centers = centers.copy()
    centers_labeled = centers.copy()
    centers_labeled[label_name] = feature
    df_violin = centers_labeled.reset_index().melt(
        id_vars=label_name,
        value_vars=centers.columns,
        value_name="p-signal",
        var_name="Cluster",
    )
    sns.violinplot(
        data=df_violin,
        x="Cluster",
        y="p-signal",
        hue=label_name,
        dodge=True,
        ax=ax,
        linewidth=0.25,
    )
    ax.legend(prop={"size": 8})
    annotation_height = df_violin["p-signal"].max() + 0.02
    for i, pval in enumerate(get_pvals_across_clusters(feature, centers)):
        if pval < 0.05:
            annotation = "*"
        elif pval < 0.01:
            annotation = "**"
        else:
            continue
        ax.text(i, annotation_height, annotation, ha="center", va="bottom", fontsize=10)


def plot_pca_on_cluster_centers(
    centers: pd.DataFrame,
    axes,
    hue_scores: np.ndarray = None,
    hue_scores_title: str = None,
    hue_loadings: np.ndarray = None,
    hue_loadings_title: str = None,
):
    # run PCA on cluster centers
    pca = PCA(n_components=2)
    scores = pca.fit_transform(centers)  # sample by PCA component
    loadings = pca.components_  # PCA component by cluster
    variance_explained = np.round(pca.explained_variance_ratio_, 2)

    # plot scores
    sns.scatterplot(
        x=scores[:, 0],
        y=scores[:, 1],
        hue=hue_scores,
        ax=axes[0],
        **{"linewidth": 0.5, "edgecolor": "k"},
    )
    if hue_scores_title:
        axes[0].legend(
            loc="lower left", prop={"size": 9}, title=hue_scores_title, fontsize=9
        )
    axes[0].set_title("PCA Scores")
    axes[0].set_xlabel(
        "PC1 (" + str(int(variance_explained[0] * 100)) + "%)", fontsize=10
    )
    axes[0].set_ylabel(
        "PC2 (" + str(int(variance_explained[1] * 100)) + "%)", fontsize=10
    )

    # plot loadings
    sns.scatterplot(
        x=loadings[0],
        y=loadings[1],
        ax=axes[1],
        hue=hue_loadings,
        **{"linewidth": 0.5, "edgecolor": "k"},
    )
    if hue_loadings_title:
        axes[1].legend(title="p < 0.01", prop={"size": 8})
    axes[1].set_title("PCA Loadings")
    axes[1].set_xlabel(
        "PC1 (" + str(int(variance_explained[0] * 100)) + "%)", fontsize=10
    )
    axes[1].set_ylabel(
        "PC2 (" + str(int(variance_explained[1] * 100)) + "%)", fontsize=10
    )
    for j, txt in enumerate(centers.columns):
        axes[1].annotate(
            txt, (loadings[0][j] + 0.001, loadings[1][j] + 0.001), fontsize=10
        )


def ExportClusterFile(cluster, cptac=False, mcf7=False):
    """Export cluster SVG file for NetPhorest and GO analysis."""
    if cptac:
        c = pd.read_csv(
            "ddmc/data/cluster_members/CPTAC_DDMC_35CL_W100_MembersCluster"
            + str(cluster)
            + ".csv"
        )
    if mcf7:
        c = pd.read_csv(
            "ddmc/data/cluster_members/msresist/data/cluster_members/CPTAC_MF7_20CL_W5_MembersCluster"
            + str(cluster)
            + ".csv"
        )
    c["pos"] = [s.split(s[0])[1].split("-")[0] for s in c["Position"]]
    c["res"] = [s[0] for s in c["Position"]]
    c.insert(4, "Gene_Human", [s + "_HUMAN" for s in c["Gene"]])
    c = c.drop(["Position"], axis=1)
    drop_list = [
        "NHSL2",
        "MAGI3",
        "SYNC",
        "LMNB2",
        "PLS3",
        "PI4KA",
        "SYNM",
        "MAP2",
        "MIA2",
        "SPRY4",
        "KSR1",
        "RUFY2",
        "MAP11",
        "MGA",
        "PRR12",
        "PCLO",
        "NCOR2",
        "BNIP3",
        "CENPF",
        "OTUD4",
        "RPA1",
        "CLU",
        "CDK18",
        "CHD1L",
        "DEF6",
        "MAST4",
        "SSR3",
    ]
    for gene in drop_list:
        c = c[c["Gene"] != gene]
    c.to_csv("Cluster_" + str(cluster) + ".csv")
