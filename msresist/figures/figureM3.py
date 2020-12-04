"""
This creates Figure M3.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy.stats import zscore, mannwhitneyu
from sklearn.linear_model import LogisticRegressionCV
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
    centers.iloc[:, :-1] = zscore(centers.iloc[:, :-1], axis=1)
    centers.columns = list(np.arange(model.ncl) + 1) + ["Patient_ID"]

    # first plot heatmap of clusters
    ax[0].axis("off")

    # PCA analysis
    centers = TumorType(centers)
    pvals = build_pval_matrix(calculate_mannW_pvals(centers), 24).iloc[:, -1].values
    plotPCA(ax[1:3], centers, 2, ["Patient_ID", "Type"], "Cluster", hue_scores="Type", style_scores="Type", pvals=pvals)

    # Plot NAT vs tumor signal per cluster
    plot_clusters_binaryfeatures(centers, "Type", ax[3])

    # Regression
    c = centers.select_dtypes(include=['float64'])
    tt = centers.iloc[:, -1]
    tt = tt.replace("Normal", 0)
    tt = tt.replace("Tumor", 1)
    lr = LogisticRegressionCV(cv=4, solver="saga", max_iter=10000, n_jobs=-1, penalty="elasticnet", class_weight="balanced", l1_ratios=[0.5, 0.9]).fit(c, tt)

    plotConfusionMatrix(ax[4], lr, c, tt)
    plotROC(ax[5], lr, c.values, tt, cv_folds=4)
    plotClusterCoefficients(ax[6], lr)

    # plot Cluster Motifs
    pssms = model.pssms(PsP_background=True)
    motifs = [pssms[9], pssms[10]]
    plotMotifs(motifs, [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], titles=["Cluster 10", "Cluster 11"], axes=ax[7:9])

    # plot Upstream Kinases
    plotUpstreamKinases(model, ax=ax[9:11], clusters_=[10, 11], n_components=4, pX=1)

    # Add subplot labels
    subplotLabel(ax)

    return f


def plot_clusters_binaryfeatures(centers, id_var, ax):
    """Plot p-signal of binary features (tumor vs NAT or mutational status) per cluster """
    ncl = centers.shape[1] - 2
    data = pd.melt(id_vars=id_var, value_vars=np.arange(ncl)+1, value_name="p-signal", var_name="Cluster", frame=centers)
    sns.stripplot(x="Cluster", y="p-signal", hue="Type", data=data, dodge=True, ax=ax)


def calculate_mannW_pvals(centers):
    """Compute Mann Whitney rank test p-values"""
    pvals = []
    for ii in range(centers.shape[1] - 2):
        x = centers.iloc[:, [ii, -1]]
        x_t = x[x["Type"] == "Normal"].iloc[:, 0]
        x_nat = x[x["Type"] == "Tumor"].iloc[:, 0]
        pval = mannwhitneyu(x_t, x_nat)[1]
        pvals.append(pval)
    return pvals


def build_pval_matrix(pvals, ncl):
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
            signif.append("Not Significant")
    data["Significant"] = signif
    return data



