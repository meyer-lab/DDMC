"""
This file contains functions that are used in multiple figures.
"""
import sys
import time
from string import ascii_uppercase
from matplotlib import gridspec, pyplot as plt
import seaborn as sns
import svgutils.transform as st
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import textwrap
import mygene
from matplotlib import gridspec, pyplot as plt
from string import ascii_uppercase
import svgutils.transform as st
import logomaker as lm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from ..clustering import DDMC
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from ..pre_processing import MeanCenter
from ..motifs import KinToPhosphotypeDict
from ..binomial import AAlist


def getSetup(figsize, gridd, multz=None):
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
    gs1 = gridspec.GridSpec(*gridd, figure=f)

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

    return (ax, f)


def subplotLabel(axs):
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
    figFile, cartoonFile, x, y, scalee=1, scale_x=1, scale_y=1, rotate=None
):
    """Add cartoon to a figure file."""

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale_x=scalee * scale_x, scale_y=scalee * scale_y)
    if rotate:
        cartoon.rotate(rotate, x, y)

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


def ComputeCenters(X, d, i, ddmc, ncl):
    """Calculate cluster centers of  different algorithms."""
    # k-means
    labels = KMeans(n_clusters=ncl).fit(d.T).labels_
    x_ = X.copy()
    x_["Cluster"] = labels
    c_kmeans = x_.groupby("Cluster").mean().T

    # GMM
    ddmc_data = DDMC(
        i,
        ncl=ncl,
        SeqWeight=0,
        distance_method=ddmc.distance_method,
        random_state=ddmc.random_state,
    ).fit(d)
    c_gmm = ddmc_data.transform()

    # DDMC seq
    ddmc_seq = DDMC(
        i,
        ncl=ncl,
        SeqWeight=ddmc.SeqWeight + 20,
        distance_method=ddmc.distance_method,
        random_state=ddmc.random_state,
    ).fit(d)
    ddmc_seq_c = ddmc_seq.transform()

    # DDMC mix
    ddmc_c = ddmc.transform()
    return [c_kmeans, c_gmm, ddmc_seq_c, ddmc_c], [
        "Unclustered",
        "k-means",
        "GMM",
        "DDMC seq",
        "DDMC mix",
    ]


def plotCenters(ax, model, xlabels, yaxis=False, drop=False):
    centers = pd.DataFrame(model.transform()).T
    centers.columns = xlabels
    if drop:
        centers = centers.drop(drop)
    num_peptides = [
        np.count_nonzero(model.labels() == jj)
        for jj in range(1, model.n_components + 1)
    ]
    for i in range(centers.shape[0]):
        cl = pd.DataFrame(centers.iloc[i, :]).T
        m = pd.melt(
            cl, value_vars=list(cl.columns), value_name="p-signal", var_name="Lines"
        )
        m["p-signal"] = m["p-signal"].astype("float64")
        sns.lineplot(
            x="Lines", y="p-signal", data=m, color="#658cbb", ax=ax[i], linewidth=2
        )
        ax[i].set_xticklabels(xlabels, rotation=45)
        ax[i].set_xticks(np.arange(len(xlabels)))
        ax[i].set_ylabel("$log_{10}$ p-signal")
        ax[i].xaxis.set_tick_params(bottom=True)
        ax[i].set_xlabel("")
        ax[i].set_title(
            "Cluster "
            + str(centers.index[i] + 1)
            + " Center "
            + "("
            + "n="
            + str(num_peptides[i])
            + ")"
        )
        if yaxis:
            ax[i].set_ylim([yaxis[0], yaxis[1]])


def plotMotifs(pssms, axes, titles=False, yaxis=False):
    """Generate logo plots of a list of PSSMs"""
    for i, ax in enumerate(axes):
        pssm = pssms[i].T
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
            logo.ax.set_title(titles[i] + " Motif")
        else:
            logo.ax.set_title("Motif Cluster " + str(i + 1))
        if yaxis:
            logo.ax.set_ylim([yaxis[0], yaxis[1]])


def plot_LassoCoef(ax, model, title=False):
    """Plot Lasso Coefficients"""
    coefs = pd.DataFrame(model.coef_).T
    coefs.index += 1
    coefs = coefs.reset_index()
    coefs.columns = ["Cluster", "Viability", "Apoptosis", "Migration", "Island"]
    m = pd.melt(
        coefs,
        id_vars="Cluster",
        value_vars=list(coefs.columns)[1:],
        var_name="Phenotype",
        value_name="Coefficient",
    )
    sns.barplot(x="Cluster", y="Coefficient", hue="Phenotype", data=m, ax=ax)
    if title:
        ax.set_title(title)


def plotDistanceToUpstreamKinase(
    model,
    clusters,
    ax,
    kind="strip",
    num_hits=5,
    additional_pssms=False,
    add_labels=False,
    title=False,
    PsP_background=True,
):
    """Plot Frobenius norm between kinase PSPL and cluster PSSMs"""
    ukin = model.predict_UpstreamKinases(
        additional_pssms=additional_pssms,
        add_labels=add_labels,
        PsP_background=PsP_background,
    )
    ukin_mc = MeanCenter(ukin, mc_col=True, mc_row=True)
    cOG = np.array(clusters).copy()
    if isinstance(add_labels, list):
        clusters += add_labels
    data = ukin_mc.sort_values(by="Kinase").set_index("Kinase")[clusters]
    if kind == "heatmap":
        sns.heatmap(data.T, ax=ax, xticklabels=data.index)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=7)
        ax.set_ylabel("Cluster")

    elif kind == "strip":
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
            print(cOG)
            AnnotateUpstreamKinases(model, list(cOG) + ["ERK2+"], ax[0], d1, 1)

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
            DrawArrows(ax[1], d2)

        else:
            sns.stripplot(data=data, x="Cluster", y="Frobenius Distance", ax=ax)
            AnnotateUpstreamKinases(model, clusters, ax, data, num_hits)
            if title:
                ax.set_title(title)


def AnnotateUpstreamKinases(model, clusters, ax, data, num_hits=1):
    """Annotate upstream kinase predictions"""
    data.iloc[:, 1] = data.iloc[:, 1].astype(str)
    pssms, _ = model.pssms()
    for ii, c in enumerate(clusters, start=1):
        cluster = data[data.iloc[:, 1] == str(c)]
        hits = cluster.sort_values(by="Frobenius Distance", ascending=True)
        hits.index = np.arange(hits.shape[0])
        hits["Phosphoacceptor"] = [KinToPhosphotypeDict[kin] for kin in hits["Kinase"]]
        try:
            cCP = pssms[c - 1].iloc[:, 5].idxmax()
        except BaseException:
            cCP == "S/T"
        if cCP == "S" or cCP == "T":
            cCP = "S/T"
        hits = hits[hits["Phosphoacceptor"] == cCP]
        for jj in range(num_hits):
            ax.annotate(
                hits["Kinase"].iloc[jj],
                (ii - 1, hits["Frobenius Distance"].iloc[jj] - 0.01),
                fontsize=8,
            )
    ax.legend().remove()
    ax.set_title("Kinase vs Cluster Motif")


def DrawArrows(ax, d2):
    data_shuff = d2[d2["Shuffled"]]
    actual_erks = d2[d2["Shuffled"] == False]
    arrow_lengths = (
        np.add(
            data_shuff["Frobenius Distance"].values,
            abs(actual_erks["Frobenius Distance"].values),
        )
        * -1
    )
    for dp in range(data_shuff.shape[0]):
        ax.arrow(
            dp,
            data_shuff["Frobenius Distance"].iloc[dp] - 0.1,
            0,
            arrow_lengths[dp] + 0.3,
            head_width=0.25,
            head_length=0.15,
            width=0.025,
            fc="black",
            ec="black",
        )


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


def build_pval_matrix(ncl, pvals):
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


def TumorType(X):
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
            "msresist/data/cluster_members/CPTAC_DDMC_35CL_W100_MembersCluster"
            + str(cluster)
            + ".csv"
        )
    if mcf7:
        c = pd.read_csv(
            "msresist/data/cluster_members/msresist/data/cluster_members/CPTAC_MF7_20CL_W5_MembersCluster"
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


def make_BPtoGenes_table(X, cluster):
    d = X[["Clusters", "Description", "geneID"]]
    d = d[d["Clusters"] == cluster]
    gAr = d[["geneID"]].values
    bpAr = d[["Description"]].values
    mg = mygene.MyGeneInfo()
    BPtoGenesDict = {}
    for ii, arr in enumerate(gAr):
        gg = mg.querymany(
            list(arr[0].split("/")),
            scopes="entrezgene",
            fields="symbol",
            species="human",
            returnall=False,
            as_dataframe=True,
        )
        BPtoGenesDict[bpAr[ii][0]] = list(gg["symbol"])
    return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in BPtoGenesDict.items()]))


def merge_binary_vectors(y, mutant1, mutant2):
    """Merge binary mutation status vectors to identify all patients having one of the two mutations"""
    y1 = y[mutant1]
    y2 = y[mutant2]
    y_ = np.zeros(y.shape[0])
    for binary in [y1, y2]:
        indices = [i for i, x in enumerate(binary) if x == 1]
        y_[indices] = 1
    return pd.Series(y_)


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
        pd.read_csv("msresist/data/CPTAC_LUAD/Hot_Cold.csv")
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
