"""
This creates Figure 5: Tumor vs NAT analysis
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from ..clustering import DDMC
from .common import subplotLabel, getSetup, plotDistanceToUpstreamKinase, TumorType, calculate_mannW_pvals, build_pval_matrix, plot_clusters_binaryfeatures
from ..logistic_regression import plotClusterCoefficients, plotROC
from ..pca import plotPCA
from ..pre_processing import filter_NaNpeptides


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((11, 10), (3, 3), multz={0: 1, 4: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Import signaling data
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=2)
    d = X.select_dtypes(include=[float]).T
    i = X.select_dtypes(include=[object])

    # Fit DDMC
    model = DDMC(i, ncl=30, SeqWeight=100, distance_method="Binomial", random_state=5).fit(d)

    # Normalize
    centers = pd.DataFrame(model.transform()).T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
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
    plotPCA(ax[1:3], centers.reset_index(), 2, ["Patient_ID", "Type"], "Cluster", hue_scores="Type", style_scores="Type", pvals=pvals.iloc[:, -1].values)
    plot_clusters_binaryfeatures(centers, "Type", ax[3], pvals=pvals, loc='lower left')

    # Transform to Binary
    c = centers.select_dtypes(include=['float64'])
    tt = centers.iloc[:, -1]
    tt = tt.replace("NAT", 0)
    tt = tt.replace("Tumor", 1)

    # Logistic Regression
    lr = LogisticRegressionCV(cv=3, solver="saga", max_iter=10000, n_jobs=-1, penalty="elasticnet", l1_ratios=[0.85], class_weight="balanced")
    plotROC(ax[4], lr, c.values, tt, cv_folds=4, return_mAUC=False)
    plotClusterCoefficients(ax[5], lr)
    ax[5].set_xticklabels(centers.columns[:-1])

    # Upstream Kinases
    plotDistanceToUpstreamKinase(model, [6, 15, 20], ax[6], num_hits=2)

    return f
