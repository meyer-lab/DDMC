"""
This file contains functions that are used in multiple figures.
"""
import sys
import time
from pathlib import Path
from string import ascii_uppercase
from matplotlib import gridspec, pyplot as plt, axes, rcParams
import seaborn as sns
import numpy as np
import pandas as pd
import svgutils.transform as st
import logomaker as lm
from sklearn.preprocessing import StandardScaler
from ..pre_processing import filter_NaNpeptides
from ..clustering import DDMC
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from ..motifs import KinToPhosphotypeDict
from ddmc.binomial import AAlist


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


def getDDMC_CPTAC(n_components: int, SeqWeight: float):
    # Import signaling data
    X = filter_NaNpeptides(
        pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:],
        tmt=2,
    )
    d = X.select_dtypes(include=[float]).T
    i = X["Sequence"]

    # Fit DDMC
    model = DDMC(
        i,
        n_components=n_components,
        SeqWeight=SeqWeight,
        distance_method="Binomial",
        random_state=5,
    ).fit(d)
    return model, X


def plot_motifs(pssm, ax: axes.Axes, titles=False, yaxis=False):
    """Generate logo plots of a list of PSSMs"""
    pssm = pssm.T
    if pssm.shape[0] == 11:
        pssm.index = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    elif pssm.shape[0] == 9:
        pssm.index = [-5, -4, -3, -2, -1, 1, 2, 3, 4]
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


def plot_cluster_kinase_distances(distances: pd.DataFrame, pssms: np.ndarray, ax, num_hits=1):
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


def plot_distance_to_upstream_kinase(
    model: DDMC,
    clusters: list[int],
    ax,
    num_hits: int = 5,
    additional_pssms=False,
    add_labels=False,
    title=False,
    PsP_background=True,
):
    """Plot Frobenius norm between kinase PSPL and cluster PSSMs"""
    ukin = model.predict_upstream_kinases(
        additional_pssms=additional_pssms,
        add_labels=add_labels,
        PsP_background=PsP_background,
    )
    ukin_mc = MeanCenter(ukin, mc_col=True, mc_row=True)
    cOG = np.array(clusters).copy()
    if isinstance(add_labels, list):
        clusters += add_labels
    data = ukin_mc.sort_values(by="Kinase").set_index("Kinase")[clusters]

    data = pd.melt(
        data.reset_index(),
        id_vars="Kinase",
        value_vars=list(data.columns),
        var_name="Cluster",
        value_name="Frobenius Distance",
    )
    if isinstance(add_labels, list):
        # Actual ERK predictions
        data["Cluster"] = data["Cluster"].astype(str)
        d1 = data[~data["Cluster"].str.contains("_S")]
        sns.stripplot(data=d1, x="Cluster", y="Frobenius Distance", ax=ax[0])
        annotate_upstream_kinases(model, list(cOG) + ["ERK2+"], ax[0], d1, 1)

        # Shuffled
        d2 = data[data["Kinase"] == "ERK2"]
        d2["Shuffled"] = ["_S" in s for s in d2["Cluster"]]
        d2["Cluster"] = [s.split("_S")[0] for s in d2["Cluster"]]
        sns.stripplot(
            data=d2,
            x="Cluster",
            y="Frobenius Distance",
            hue="Shuffled",
            ax=ax[1],
            size=8,
        )
        ax[1].set_title("ERK2 Shuffled Positions")
        ax[1].legend(prop={"size": 10}, loc="lower left")
        draw_arrows(ax[1], d2)
    else:
        sns.stripplot(data=data, x="Cluster", y="Frobenius Distance", ax=ax)
        annotate_upstream_kinases(model, clusters, ax, data, num_hits)
        if title:
            ax.set_title(title)


def plot_clusters_binaryfeatures(centers, id_var, ax, pvals=False, loc="best"):
    """Plot p-signal of binary features (tumor vs NAT or mutational status) per cluster"""
    data = pd.melt(
        id_vars=id_var,
        value_vars=centers.columns[:-1],
        value_name="p-signal",
        var_name="Cluster",
        frame=centers,
    )
    sns.violinplot(
        x="Cluster",
        y="p-signal",
        hue=id_var,
        data=data,
        dodge=True,
        ax=ax,
        linewidth=0.25,
        fliersize=2,
    )
    ax.legend(prop={"size": 8}, loc=loc)

    if not isinstance(pvals, bool):
        for ii, s in enumerate(pvals["Significant"]):
            y, h, col = data["p-signal"].max(), 0.05, "k"
            if s == "NS":
                continue
            elif s == "<0.05":
                mark = "*"
            else:
                mark = "**"
            ax.text(ii, y + h, mark, ha="center", va="bottom", color=col, fontsize=20)


def calculate_mannW_pvals(centers, col, feature1, feature2):
    """Compute Mann Whitney rank test p-values corrected for multiple tests."""
    pvals, clus = [], []
    for ii in centers.columns[:-1]:
        x = centers.loc[:, [ii, col]]
        x1 = x[x[col] == feature1].iloc[:, 0]
        x2 = x[x[col] == feature2].iloc[:, 0]
        pval = mannwhitneyu(x1, x2)[1]
        pvals.append(pval)
        clus.append(ii)
    return dict(zip(clus, multipletests(pvals)[1]))


def build_pval_matrix(ncl, pvals) -> pd.DataFrame:
    """Build data frame with pvalues per cluster"""
    data = pd.DataFrame()
    data["Clusters"] = pvals.keys()
    data["p-value"] = pvals.values()
    signif = []
    for val in pvals.values():
        if 0.01 < val < 0.05:
            signif.append("<0.05")
        elif val < 0.01:
            signif.append("<0.01")
        else:
            signif.append("NS")
    data["Significant"] = signif
    return data


def TumorType(X: pd.DataFrame) -> pd.DataFrame:
    """Add NAT vs Tumor column."""
    tumortype = []
    for i in range(X.shape[0]):
        if X["Patient_ID"][i].endswith(".N"):
            tumortype.append("NAT")
        else:
            tumortype.append("Tumor")
    X["Type"] = tumortype
    return X


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


def find_patients_with_NATandTumor(X, label, conc=False):
    """Reshape data to display patients as rows and samples (Tumor and NAT per cluster) as columns.
    Note that to do so, samples that don't have their tumor/NAT counterpart are dropped.
    """
    xT = X[~X[label].str.endswith(".N")].sort_values(by=label)
    xN = X[X[label].str.endswith(".N")].sort_values(by=label)
    l1 = list(xT[label])
    l2 = [s.split(".N")[0] for s in xN[label]]
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    X = xT.set_index(label).drop(dif)
    assert all(X.index.values == np.array(l2)), "Samples don't match"

    if conc:
        xN = xN.set_index(label)
        xN.index = l2
        xN.columns = [str(i) + "_Normal" for i in xN.columns]
        X.columns = [str(i) + "_Tumor" for i in X.columns]
        X = pd.concat([X, xN], axis=1)
    return X


def TransformCenters(model, X):
    """For a given model, find centers and transform for regression."""
    centers = pd.DataFrame(model.transform()).T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(
        centers.iloc[:, :]
    )
    centers = centers.T
    centers.columns = np.arange(model.n_components) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers1 = find_patients_with_NATandTumor(centers.copy(), "Patient_ID", conc=True)
    centers2 = (
        centers.loc[~centers["Patient_ID"].str.endswith(".N"), :]
        .sort_values(by="Patient_ID")
        .set_index("Patient_ID")
    )
    return centers1, centers2


def HotColdBehavior(centers):
    # Import Cold-Hot Tumor data
    y = (
        pd.read_csv("ddmc/data/MS/CPTAC/Hot_Cold.csv")
        .dropna(axis=1)
        .sort_values(by="Sample ID")
    )
    y = y.loc[~y["Sample ID"].str.endswith(".N"), :].set_index("Sample ID")
    l1 = list(centers.index)
    l2 = list(y.index)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    centers = centers.drop(dif)

    # Transform to binary
    y = y.replace("Cold-tumor enriched", 0)
    y = y.replace("Hot-tumor enriched", 1)
    y = np.squeeze(y)

    # Remove NAT-enriched samples
    centers = centers.drop(y[y == "NAT enriched"].index)
    y = y.drop(y[y == "NAT enriched"].index).astype(int)
    assert all(centers.index.values == y.index.values), "Samples don't match"

    return y, centers
