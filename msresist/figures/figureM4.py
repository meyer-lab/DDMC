"""
This creates Figure 3: Tumor vs NAT analysis
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import textwrap
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from .common import subplotLabel, getSetup
from ..logistic_regression import plotClusterCoefficients, plotROC
from ..figures.figure2 import plotPCA, plotMotifs, plotDistanceToUpstreamKinase
from ..pre_processing import MeanCenter


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((14, 9), (3, 4), multz={0: 1, 4: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    import pandas as pd
    from msresist.clustering import MassSpecClustering
    from msresist.validations import preprocess_ebdt_mcf7

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]

    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    # first plot heatmap of clusters
    ax[0].axis("off")

    # Normalize
    centers = pd.DataFrame(model.transform()).T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    centers = centers.T
    centers.columns = np.arange(model.ncl) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = TumorType(centers).set_index("Patient_ID")
    centers["Type"] = centers["Type"].replace("Normal", "NAT")

    # PCA and Hypothesis Testing
    pvals = calculate_mannW_pvals(centers, "Type", "NAT", "Tumor")
    pvals = build_pval_matrix(model.ncl, pvals)
    plotPCA(ax[1:3], centers.reset_index(), 2, ["Patient_ID", "Type"], "Cluster", hue_scores="Type", style_scores="Type", pvals=pvals.iloc[:, -1].values)
    plot_clusters_binaryfeatures(centers, "Type", ax[3], pvals=pvals, loc='lower left')

    # Transform to Binary
    c = centers.select_dtypes(include=['float64'])
    tt = centers.iloc[:, -1]
    tt = tt.replace("NAT", 0)
    tt = tt.replace("Tumor", 1)

    # Logistic Regression
    lr = LogisticRegressionCV(Cs=10, cv=10, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")
    plotROC(ax[4], lr, c.values, tt, cv_folds=4, return_mAUC=False)
    plotClusterCoefficients(ax[5], lr)

    # plot Upstream Kinases
    plotDistanceToUpstreamKinase(model, [11, 12], ax[6], num_hits=3)

    # Cluster 11
    plot_GO(11, ax[7], n=5, title="GO Cluster 11")

    # Cluster 12
    plot_GO(12, ax[8], n=3, title="GO Cluster 12")

    plot_NetPhoresScoreByKinGroup("msresist/data/cluster_analysis/CPTAC_NK_C12.csv", ax[9], n=5, title="Cluster 12 Kinase Predictions")

    return f


def plot_NetPhoresScoreByKinGroup(PathToFile, ax, n=5, title=False):
    """Plot top scoring kinase groups"""
    NPtoCumScore = {}
    X = pd.read_csv(PathToFile)
    for ii in range(X.shape[0]):
        curr_NPgroup = X["netphorest_group"][ii]
        if curr_NPgroup not in NPtoCumScore.keys():
            NPtoCumScore[curr_NPgroup] = X["netphorest_score"][ii]
        else:
            NPtoCumScore[curr_NPgroup] += X["netphorest_score"][ii]
    X = pd.DataFrame.from_dict(NPtoCumScore, orient='index').reset_index()
    X.columns = ["KIN Group", "NetPhorest Score"]
    X["KIN Group"] = [s.split("_")[0] for s in X["KIN Group"]]
    X = X.sort_values(by="NetPhorest Score", ascending=False).iloc[:n, :]
    sns.barplot(data=X, y="KIN Group", x="NetPhorest Score", ax=ax, orient="h", color="darkblue", **{"linewidth": 2}, **{"edgecolor": "black"})
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Kinase Predictions")


def plot_GO(cluster, ax, n=5, title=False, max_width=25):
    """Plot top scoring gene ontologies in a cluster"""
    X = pd.read_csv("msresist/data/cluster_analysis/CPTAC_GO_C" + str(cluster) + ".csv")
    X = X[["GO biological process complete", "upload_1 (fold Enrichment)"]].iloc[:n, :]
    X.columns = ["Biological process", "Fold Enrichment"]
    X["Fold Enrichment"] = X["Fold Enrichment"].astype(float)
    X["Biological process"] = [s.split("(GO")[0] for s in X["Biological process"]]
    sns.barplot(data=X, y="Biological process", x="Fold Enrichment", ax=ax, orient="h", color="lightgrey", **{"linewidth": 2}, **{"edgecolor": "black"})
    ax.set_yticklabels(textwrap.fill(x.get_text(), max_width) for x in ax.get_yticklabels())
    if title:
        ax.set_title(title)
    else:
        ax.set_title("GO")


def TumorType(X):
    """Add Normal vs Tumor column."""
    tumortype = []
    for i in range(X.shape[0]):
        if ".N" in X["Patient_ID"][i]:
            tumortype.append("Normal")
        else:
            tumortype.append("Tumor")
    X["Type"] = tumortype
    return X


def plot_clusters_binaryfeatures(centers, id_var, ax, pvals=False, loc='best'):
    """Plot p-signal of binary features (tumor vs NAT or mutational status) per cluster """
    ncl = centers.shape[1] - 1
    data = pd.melt(id_vars=id_var, value_vars=np.arange(ncl) + 1, value_name="p-signal", var_name="Cluster", frame=centers)
    sns.violinplot(x="Cluster", y="p-signal", hue=id_var, data=data, dodge=True, ax=ax, linewidth=0.5, fliersize=2)
    ax.legend(prop={'size': 8}, loc=loc)

    if not isinstance(pvals, bool):
        for ii, s in enumerate(pvals["Significant"]):
            y, h, col = data['p-signal'].max(), .05, 'k'
            if s == "NS":
                continue
            elif s == "<0.05":
                mark = "*"
            else:
                mark = "**"
            ax.text(ii, y + h, mark, ha='center', va='bottom', color=col, fontsize=20)


def calculate_mannW_pvals(centers, col, feature1, feature2):
    """Compute Mann Whitney rank test p-values"""
    pvals = []
    for ii in range(centers.shape[1] - 1):
        x = centers.iloc[:, [ii, -1]]
        x1 = x[x[col] == feature1].iloc[:, 0]
        x2 = x[x[col] == feature2].iloc[:, 0]
        pval = mannwhitneyu(x1, x2)[1]
        pvals.append(pval)
    pvals = multipletests(pvals)[1]  # p-value correction for multiple tests
    return pvals


def build_pval_matrix(ncl, pvals):
    """Build data frame with pvalues per cluster"""
    data = pd.DataFrame()
    data["Clusters"] = np.arange(ncl) + 1
    data["p-value"] = pvals
    signif = []
    for val in pvals:
        if 0.01 < val < 0.05:
            signif.append("<0.05")
        elif val < 0.01:
            signif.append("<0.01")
        else:
            signif.append("NS")
    data["Significant"] = signif
    return data


def ExportClusterFile(cluster):
    """Export cluster SVG file for NetPhorest and GO analysis."""
    c = pd.read_csv("msresist/data/cluster_members/CPTACmodel_Members_C" + str(cluster) + ".csv")
    c["pos"] = [s.split(s[0])[1].split("-")[0] for s in c["Position"]]
    c["res"] = [s[0] for s in c["Position"]]
    c.insert(4, "Gene_Human", [s + "_HUMAN" for s in c["Gene"]])
    c = c.drop(["Position", "Cluster"], axis=1)
    drop_list = ["NHSL2", "MAGI3", "SYNC", "LMNB2", "PLS3", "PI4KA", "SYNM", "MAP2", "MIA2", "SPRY4", "KSR1", "RUFY2", "MAP11",
                 "MGA", "PRR12", "PCLO", "NCOR2", "BNIP3", "CENPF", "OTUD4", "RPA1", "CLU", "CDK18", "CHD1L", "DEF6", "MAST4", "SSR3"]
    for gene in drop_list:
        c = c[c["Gene"] != gene]
    c.to_csv("Cluster_" + str(cluster) + ".csv")
