"""
This creates Supplemental Figure 4.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from pomegranate import GeneralMixtureModel, NormalDistribution
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from ..logistic_regression import plotClusterCoefficients, plotConfusionMatrix, plotROC
from .figureM3 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals
from .figureMS3 import plot_unclustered_LRcoef


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 20), (5, 2), multz={4: 1, 8: 1})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Add subplot labels
    subplotLabel(ax)

    # STK11 WT vs mut unclustered
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, cut=1)
    d = X.set_index("Gene").select_dtypes(include=["float64"]).T.reset_index()
    d.rename(columns={"index": "Patient_ID"}, inplace=True)

    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    mOI = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = mOI.set_index("Sample.ID")
    y = y["STK11.mutation.status"]

    # Remove NATs
    X = X.loc[:, ~X.columns.str.endswith(".N")]
    d = d[~d["Patient_ID"].str.endswith(".N")].iloc[:, 1:]
    y = y[~y.index.str.endswith(".N")]
    z = d.copy()
    z["STK11 status"] = y.values

    svc = LinearSVC(penalty="l1", dual=False, max_iter=10000, tol=1e-7)
    uc_svc = svc.fit(d, y)

    plotROC(ax[0], uc_svc, d.values, y, cv_folds=4, title="ROC unclustered")
    plot_unclustered_LRcoef(ax[1], uc_svc, z)

    # STK11 WT vs mut k-means
    ncl = 24
    labels = KMeans(n_clusters=ncl).fit(d.T).labels_
    x_ = X.copy()
    x_["Cluster"] = labels
    c_kmeans = x_.groupby("Cluster").mean().T
    c_kmeans.columns = list(np.arange(ncl) + 1)
    km_svc = svc.fit(c_kmeans, y)
    plotROC(ax[2], km_svc, c_kmeans.values, y, cv_folds=4, title="ROC k-means")
    plotClusterCoefficients(ax[3], svc, title="k-means")
    c_kmeans["STK11 status"] = z.iloc[:, -1].values
    pvals = calculate_mannW_pvals(c_kmeans, "STK11 status", 0, 1)
    pvals = build_pval_matrix(ncl, pvals)
    plot_clusters_binaryfeatures(c_kmeans, "STK11 status", ax[4], pvals=pvals)

    # Tumor vs NAT GMM
    ncl = 15
    for _ in range(10):
        gmm = GeneralMixtureModel.from_samples(NormalDistribution, X=d.T, n_components=ncl, n_jobs=-1)
        scores = gmm.predict_proba(d.T)
        if np.all(np.isfinite(scores)):
            break
    x_ = X.copy()
    x_["Cluster"] = gmm.predict(d.T)
    c_gmm = x_.groupby("Cluster").mean().T
    c_gmm.columns = list(np.arange(ncl) + 1)
    gmm_svc = svc.fit(c_gmm, y)
    plotROC(ax[5], gmm_svc, c_gmm.values, y, cv_folds=4, title="ROC GMM")
    plotClusterCoefficients(ax[6], gmm_svc, title="GMM")
    c_gmm["STK11 status"] = z.iloc[:, -1].values
    pvals = calculate_mannW_pvals(c_gmm, "STK11 status", 0, 1)
    pvals = build_pval_matrix(ncl, pvals)
    plot_clusters_binaryfeatures(c_gmm, "STK11 status", ax[7], pvals=pvals)

    return f
