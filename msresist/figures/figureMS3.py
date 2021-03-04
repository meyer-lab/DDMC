"""
This creates Supplemental Figure 3: Predicting sample type with different modeling strategies
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from pomegranate import GeneralMixtureModel, NormalDistribution
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from ..logistic_regression import plotClusterCoefficients, plotROC
from .figureMS6 import TumorType
from .figureM3 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((8, 12), (5, 2), multz={4: 1, 8: 1})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Add subplot labels
    subplotLabel(ax)

    # Tumor vs NAT unclustered
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, cut=1)
    X["Gene/Pos"] = X["Gene"] + ": " + X["Position"]
    d = X.set_index("Gene/Pos").select_dtypes(include=["float64"]).T.reset_index()
    d.rename(columns={"index": "Patient_ID"}, inplace=True)
    z = TumorType(d)
    z.iloc[:, -1] = z.iloc[:, -1].replace("Normal", "NAT")
    d = z.iloc[:, 1:-1]
    y = z.iloc[:, -1]
    y = y.replace("NAT", 0)
    y = y.replace("Tumor", 1)

    lr = LogisticRegressionCV(Cs=10, cv=10, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")
    plotROC(ax[0], lr, d.values, y, cv_folds=4, title="ROC unclustered")
    plot_unclustered_LRcoef(ax[1], lr.fit(d, y), z, title="p-sites explaining tumor vs NATs")

    # Tumor vs NAT k-means
    ncl = 24
    labels = KMeans(n_clusters=ncl).fit(d.T).labels_
    x_ = X.copy()
    x_["Cluster"] = labels
    c_kmeans = x_.groupby("Cluster").mean().T
    c_kmeans.columns = list(np.arange(ncl) + 1)
    km_lr = lr.fit(c_kmeans, y)
    plotROC(ax[2], km_lr, c_kmeans.values, y, cv_folds=4, title="ROC k-means")
    plotClusterCoefficients(ax[3], lr, title="k-means")
    c_kmeans["Type"] = z.iloc[:, -1].values
    pvals = calculate_mannW_pvals(c_kmeans, "Type", "NAT", "Tumor")
    pvals = build_pval_matrix(ncl, pvals)
    plot_clusters_binaryfeatures(c_kmeans, "Type", ["Tumor", "NAT"], ax[4], pvals=pvals)

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
    gmm_lr = lr.fit(c_gmm, y)
    plotROC(ax[5], gmm_lr, c_gmm.values, y, cv_folds=4, title="ROC GMM")
    plotClusterCoefficients(ax[6], gmm_lr, title="GMM")
    c_gmm["Type"] = z.iloc[:, -1].values
    pvals = calculate_mannW_pvals(c_gmm, "Type", "NAT", "Tumor")
    pvals = build_pval_matrix(ncl, pvals)
    plot_clusters_binaryfeatures(c_gmm, "Type", ["Tumor", "NAT"], ax[7], pvals=pvals)

    return f


def plot_unclustered_LRcoef(ax, lr, d, title=False):
    """Plot logistic regression coefficients of unclustered data"""
    cdic = dict(zip(lr.coef_[0], d.columns))
    coefs = pd.DataFrame()
    coefs["Coefficients"] = list(cdic.keys())
    coefs["p-sites"] = list(cdic.values())
    coefs.sort_values(by="Coefficients", ascending=False, inplace=True)
    sliced_coefs = coefs.head(5)
    coefs = sliced_coefs.append(coefs.tail(5))
    sns.barplot(data=coefs, x="p-sites", y="Coefficients", ax=ax, color="darkblue")
    ax.set_title(title)
    ax.set_xticklabels(list(set(coefs["p-sites"])), rotation=90)
