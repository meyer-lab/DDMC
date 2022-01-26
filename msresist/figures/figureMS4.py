"""
This creates Supplemental Figure 4: Predicting sample type with different modeling strategies
"""

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from ..clustering import DDMC
from .common import subplotLabel, getSetup, TumorType
from ..pre_processing import filter_NaNpeptides
from ..logistic_regression import plotROC


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 6), (2, 3), multz={1: 1})

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Add subplot labels
    subplotLabel(ax)

    # Import data
    X = pd.read_csv("msresist/data/CPTAC_LUAD/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, cut=1)
    i = X.select_dtypes(include=['object'])
    X["Gene/Pos"] = X["Gene"] + ": " + X["Position"]
    d = X.set_index("Gene/Pos").select_dtypes(include=["float64"]).T.reset_index()
    d.rename(columns={"index": "Patient_ID"}, inplace=True)
    z = TumorType(d)
    z.iloc[:, -1] = z.iloc[:, -1].replace("Normal", "NAT")
    d = z.iloc[:, 1:-1]
    y = z.iloc[:, -1]
    y = y.replace("NAT", 0)
    y = y.replace("Tumor", 1)

    # DDMC ROC
    ncl = 30
    model = DDMC(i, ncl=ncl, SeqWeight=100, distance_method="Binomial", random_state=5).fit(d)
    lr = LogisticRegressionCV(cv=3, solver="saga", max_iter=10000, n_jobs=-1, penalty="elasticnet", l1_ratios=[0.85], class_weight="balanced")
    plotROC(ax[2], lr, model.transform(), y, cv_folds=4, return_mAUC=False)
    ax[2].set_title("DDMC ROC")

    # Tumor vs NAT unclustered
    plotROC(ax[0], lr, d.values, y, cv_folds=4, title="ROC unclustered")
    ax[0].set_title("Unclustered ROC")
    plot_unclustered_LRcoef(ax[1], lr, d, y, z)

    # k-means
    labels = KMeans(n_clusters=ncl).fit(d.T).labels_
    x_ = X.copy()
    x_["Cluster"] = labels
    c_kmeans = x_.groupby("Cluster").mean().T
    c_kmeans.columns = list(np.arange(ncl) + 1)
    km_lr = lr.fit(c_kmeans, y)
    plotROC(ax[3], km_lr, c_kmeans.values, y, cv_folds=4, title="ROC k-means")
    ax[3].set_title("k-means ROC")

    # GMM
    gmm = DDMC(i, ncl=ncl, SeqWeight=0, distance_method="Binomial").fit(d)
    x_ = X.copy()
    x_["Cluster"] = gmm.labels()
    c_gmm = x_.groupby("Cluster").mean().T
    gmm_lr = lr.fit(c_gmm, y)
    plotROC(ax[4], gmm_lr, c_gmm.values, y, cv_folds=4, title="ROC GMM")
    ax[4].set_title("GMM ROC")

    return f


def plot_unclustered_LRcoef(ax, lr, d, y, z, title=False):
    """Plot logistic regression coefficients of unclustered data"""
    weights = []
    w = pd.DataFrame()
    for _ in range(3):
        lr = LogisticRegressionCV(cv=3, solver="saga", max_iter=10000, n_jobs=-1, penalty="elasticnet", l1_ratios=[0.85], class_weight="balanced")
        w["Coefficients"] = lr.fit(d, y).coef_[0]
        w["p-sites"] = z.columns[2:]
        weights.append(w)

    coefs = pd.concat(weights)
    coefs.sort_values(by="Coefficients", ascending=False, inplace=True)
    coefs = coefs[(coefs["Coefficients"] > 0.075) | (coefs["Coefficients"] < -0.075)]
    sns.barplot(data=coefs, x="p-sites", y="Coefficients", ax=ax, color="darkblue")
    ax.set_title("p-sites explaining tumor vs NATs ")
    ax.set_xticklabels(list(set(coefs["p-sites"])), rotation=90)
    ax.set_xlabel("p-sites")
    return coefs
