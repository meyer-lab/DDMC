"""
This creates Figure 5: Tumor vs NAT analysis
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import textwrap
import mygene
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from ..clustering import DDMC
from .common import subplotLabel, getSetup
from ..logistic_regression import plotClusterCoefficients, plotROC
from .common import plotDistanceToUpstreamKinase
from ..pca import plotPCA
from ..pre_processing import filter_NaNpeptides


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((11, 10), (3, 3), multz={0: 1, 4: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams["font.sans-serif"] = "Arial"
    sns.set(
        style="whitegrid",
        font_scale=1,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # Import signaling data
    X = filter_NaNpeptides(
        pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:],
        tmt=2,
    )
    d = X.select_dtypes(include=[float]).T
    i = X["Sequence"]

    # Fit DDMC
    model = DDMC(
        i, n_components=30, SeqWeight=100, distance_method="Binomial", random_state=5
    ).fit(d)

    # Normalize
    centers = pd.DataFrame(model.transform()).T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(
        centers.iloc[:, :]
    )
    centers = centers.T
    centers.columns = np.arange(model.n_components) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = TumorType(centers).set_index("Patient_ID")
    centers = centers.drop([14, 24], axis=1)  # Drop cluster 19, contains only 1 peptide

    # first plot heatmap of clusters
    # lim = 1.5
    # sns.clustermap(centers.set_index("Type").T, method="complete", cmap="bwr", vmax=lim, vmin=-lim,  figsize=(15, 9)) Run in notebook and save as svg
    ax[0].axis("off")

    # PCA and Hypothesis Testing
    pvals = calculate_mannW_pvals(centers, "Type", "NAT", "Tumor")
    pvals = build_pval_matrix(model.n_components, pvals)
    plotPCA(
        ax[1:3],
        centers.reset_index(),
        2,
        ["Patient_ID", "Type"],
        "Cluster",
        hue_scores="Type",
        style_scores="Type",
        pvals=pvals.iloc[:, -1].values,
    )
    plot_clusters_binaryfeatures(centers, "Type", ax[3], pvals=pvals, loc="lower left")

    # Transform to Binary
    c = centers.select_dtypes(include=["float64"])
    tt = centers.iloc[:, -1]
    tt = tt.replace("NAT", 0)
    tt = tt.replace("Tumor", 1)

    # Logistic Regression
    lr = LogisticRegressionCV(
        cv=3,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="elasticnet",
        l1_ratios=[0.85],
        class_weight="balanced",
    )
    plotROC(ax[4], lr, c.values, tt, cv_folds=4, return_mAUC=False)
    plotClusterCoefficients(ax[5], lr)
    ax[5].set_xticklabels(centers.columns[:-1])

    # Upstream Kinases
    plotDistanceToUpstreamKinase(model, [6, 15, 20], ax[6], num_hits=2)

    return f


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


def plot_enriched_processes(ax, X, y, f, cluster, gene_set="WP"):
    """"Plot BPs enriched per cluster""" ""
    if gene_set == "WP":
        gsea = pd.read_csv("ddmc/data/cluster_analysis/CPTAC_GSEA_WP_results.csv").iloc[
            :, 1:
        ]
    elif gene_set == "onco":
        gsea = pd.read_csv(
            "ddmc/data/cluster_analysis/CPTAC_GSEA_ONCO_results.csv"
        ).iloc[:, 1:]
    elif gene_set == "Immuno":
        gsea = pd.read_csv("ddmc/data/cluster_analysis/CPTAC_GSEA_WP_results.csv").iloc[
            :, 1:
        ]
    cc = make_BPtoGenes_table(gsea, cluster)
    cl = X[X["Cluster"] == cluster].set_index("Gene")
    dfs = []
    for ii in range(cc.shape[1]):
        ss = cl.loc[cc.iloc[:, ii].dropna()].reset_index()
        ss["Process"] = cc.columns[ii]
        dfs.append(ss)

    out = pd.concat(dfs).set_index("Process").select_dtypes(include=[float]).T
    out[f[0]] = y
    out[f[0]] = out[f[0]].replace(0, f[1])
    out[f[0]] = out[f[0]].replace(1, f[2])
    dm = pd.melt(
        out,
        id_vars=f[0],
        value_vars=out.columns,
        var_name="Process",
        value_name="mean log(p-signal)",
    )
    dm.iloc[:, -1] = dm.iloc[:, -1].astype(float)
    sns.boxplot(
        data=dm,
        x="Process",
        y="mean log(p-signal)",
        hue=f[0],
        showfliers=False,
        linewidth=0.5,
        ax=ax,
    )
    ax.set_xticklabels([textwrap.fill(t, 10) for t in list(cc.columns)], rotation=0)
    ax.set_title("Processes Cluster " + str(cluster))


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
