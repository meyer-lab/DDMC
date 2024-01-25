"""
This creates Supplemental Figure 4: Predicting sample type with different modeling strategies
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
from ..clustering import DDMC
from .common import getSetup
from ..pre_processing import filter_NaNpeptides
from ..logistic_regression import plotROC
from .figureM5 import TumorType


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 6), (2, 3), multz={1: 1})

    # Import data
    X = pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, cut=1)
    X["Gene/Pos"] = X["Gene"] + ": " + X["Position"]
    d = X.set_index("Gene/Pos").select_dtypes(include=["float64"]).T.reset_index()
    d.rename(columns={"index": "Patient_ID"}, inplace=True)
    z = TumorType(d)
    z.iloc[:, -1] = z.iloc[:, -1].replace("Normal", "NAT")
    d = z.iloc[:, 1:-1]

    y = label_binarize(z["Type"], classes=["NAT", "Tumor"])

    # DDMC ROC
    ncl = 30
    model = DDMC(
        X["Sequence"],
        n_components=ncl,
        seq_weight=100,
        distance_method="Binomial",
        random_state=5,
    ).fit(d)
    lr = LogisticRegressionCV(
        cv=3,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="elasticnet",
        l1_ratios=[0.85],
        class_weight="balanced",
    )
    plotROC(ax[2], lr, model.transform(), y, cv_folds=4, return_mAUC=False)
    ax[2].set_title("DDMC ROC")

    # Tumor vs NAT unclustered
    plotROC(ax[0], lr, d.values, y, cv_folds=4, title="ROC unclustered")
    ax[0].set_title("Unclustered ROC")
    plot_unclustered_LRcoef(ax[1], lr, d, y)

    # k-means
    kmeans = KMeans(n_clusters=ncl).fit(d.T)

    plotROC(ax[3], lr, kmeans.cluster_centers_.T, y, cv_folds=4, title="ROC k-means")
    ax[3].set_title("k-means ROC")

    # GMM
    gmm = DDMC(
        X["Sequence"], n_components=ncl, seq_weight=0, distance_method="Binomial"
    ).fit(d)

    plotROC(ax[4], lr, gmm.transform(), y, cv_folds=4, title="ROC GMM")
    ax[4].set_title("GMM ROC")

    return f


def plot_unclustered_LRcoef(ax, lr, X: pd.DataFrame, y: np.ndarray):
    """Plot logistic regression coefficients of unclustered data"""
    weights = []
    w = pd.DataFrame()
    for _ in range(3):
        w["Coefficients"] = lr.fit(X, y).coef_[0]
        w["p-sites"] = X.columns[2:]
        weights.append(w)

    coefs = pd.concat(weights)
    coefs.sort_values(by="Coefficients", ascending=False, inplace=True)
    coefs = coefs[(coefs["Coefficients"] > 0.075) | (coefs["Coefficients"] < -0.075)]
    sns.barplot(data=coefs, x="p-sites", y="Coefficients", ax=ax, color="darkblue")
    ax.set_title("p-sites explaining tumor vs NATs ")
    ax.set_xticklabels(list(set(coefs["p-sites"])), rotation=90)
    ax.set_xlabel("p-sites")
    return coefs
