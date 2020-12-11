"""
This creates Figure M3.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy.stats import zscore, mannwhitneyu
from sklearn.linear_model import LogisticRegressionCV
from statsmodels.stats.multitest import multipletests
from .common import subplotLabel, getSetup
from ..figures.figureM2 import TumorType
from ..logistic_regression import plotClusterCoefficients, plotConfusionMatrix, plotROC
from ..figures.figure3 import plotPCA, plotMotifs, plotUpstreamKinases
from ..clustering import MassSpecClustering
from ..pre_processing import filter_NaNpeptides, MeanCenter
import pickle


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((13, 15), (4, 3), multz={3: 1})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]

    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    centers = pd.DataFrame(model.transform())
    centers["Patient_ID"] = X.columns[4:]
    centers.columns = list(np.arange(model.ncl) + 1) + ["Patient_ID"]

    # first plot heatmap of clusters
    ax[0].axis("off")

    # PCA analysis
    centers = TumorType(centers)
    pvals = build_pval_matrix(model.ncl, centers, "Type", "Normal", "Tumor").iloc[:, -1].values
    c_ = centers.copy()
    c_.iloc[:, :-2] = zscore(c_.iloc[:, :-2], axis=1)  # zscore for PCA
    plotPCA(ax[1:3], c_, 2, ["Patient_ID", "Type"], "Cluster", hue_scores="Type", style_scores="Type", pvals=pvals)

    # Plot NAT vs tumor signal per cluster
    plot_clusters_binaryfeatures(centers, "Type", ax[3])

    # Regression
    c = c_.select_dtypes(include=['float64'])
    tt = centers.iloc[:, -1]
    tt = tt.replace("Normal", 0)
    tt = tt.replace("Tumor", 1)
    lr = LogisticRegressionCV(cv=4, solver="saga", max_iter=10000, n_jobs=-1, penalty="elasticnet", class_weight="balanced", l1_ratios=[0.5, 0.9]).fit(c, tt)

    plotConfusionMatrix(ax[4], lr, c, tt)
    plotROC(ax[5], lr, c.values, tt, cv_folds=4)
    plotClusterCoefficients(ax[6], lr)

    # plot Cluster Motifs
    pssms = model.pssms(PsP_background=False)
    motifs = [pssms[10], pssms[11]]
    plotMotifs(motifs, [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], titles=["Cluster 11", "Cluster 12"], axes=ax[7:9])

    # plot Upstream Kinases
    plotUpstreamKinases(model, ax=ax[9:11], clusters_=[11, 12], n_components=4, pX=1)

    # Add subplot labels
    subplotLabel(ax)

    return f


def plot_clusters_binaryfeatures(centers, id_var, ax, pvals=False):
    """Plot p-signal of binary features (tumor vs NAT or mutational status) per cluster """
    ncl = centers.shape[1] - 2
    data = pd.melt(id_vars=id_var, value_vars=np.arange(ncl) + 1, value_name="p-signal", var_name="Cluster", frame=centers)
    sns.stripplot(x="Cluster", y="p-signal", hue=id_var, data=data, dodge=True, ax=ax, alpha=0.4)
    sns.boxplot(x="Cluster", y="p-signal", hue=id_var, data=data, dodge=True, ax=ax, color="white", linewidth=2)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(title="TP53 status", labels=["WT", "mut"], handles=handles[2:])
    if not isinstance(pvals, bool):
        for ii, s in enumerate(pvals["Significant"]):
            y, h, col = data['p-signal'].max(), .05, 'k'
            if s == "Not Significant":
                continue
            elif s == "<0.05":
                mark = "*"
            else:
                mark = "**"
            ax.text(ii, y + h, mark, ha='center', va='bottom', color=col, fontsize=20)


def calculate_mannW_pvals(centers, col, feature1, feature2):
    """Compute Mann Whitney rank test p-values"""
    pvals = []
    for ii in range(centers.shape[1] - 2):
        x = centers.iloc[:, [ii, -1]]
        x1 = x[x[col] == feature1].iloc[:, 0]
        x2 = x[x[col] == feature2].iloc[:, 0]
        pval = mannwhitneyu(x1, x2)[1]
        pvals.append(pval)
    pvals = multipletests(pvals)[1]  # p-value correction for multiple tests
    return pvals


def build_pval_matrix(ncl, centers, col, feature1, feature2):
    """Build data frame with pvalues per cluster"""
    pvals = calculate_mannW_pvals(centers, col, feature1, feature2)
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
            signif.append("Not Significant")
    data["Significant"] = signif
    return data
