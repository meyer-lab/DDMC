"""
This creates Supplemental Figure 7: Predicting STK11 genotype using different clustering strategies.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ddmc.clustering import DDMC
from .common import getSetup
from ..pre_processing import filter_NaNpeptides
from ..logistic_regression import plotROC
from .figureM4 import find_patients_with_NATandTumor


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (2, 5))

    # Signaling
    X = filter_NaNpeptides(
        pd.read_csv("ddmc/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:],
        cut=1,
    )

    # Fit DDMC to complete data
    d = np.array(X.select_dtypes(include=["float64"]).T)
    i = X["Sequence"]

    assert np.all(np.isfinite(d))
    model_min = DDMC(i, n_components=30, SeqWeight=100, distance_method="Binomial").fit(
        d
    )

    centers_min = reshapePatients(model_min.transform(), X.columns[4:])

    # Fit DDMC
    model = DDMC(i, n_components=30, SeqWeight=100, distance_method="Binomial").fit(d)

    centers = reshapePatients(model.transform(), X.columns[4:])

    # Predicting STK11
    lr = LogisticRegressionCV(
        cv=5,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="l1",
        class_weight="balanced",
        random_state=10,
    )
    mutations = pd.read_csv("ddmc/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[
        ["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]
    ]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)
    assert all(centers.index.values == y.index.values), "Samples don't match"
    y_STK = y["STK11.mutation.status"]
    plot_ROCs(ax[:5], centers, centers_min, X, i, y_STK, lr, "STK11")

    # Predicting EGFRm
    lr = LogisticRegressionCV(
        cv=20,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="l1",
        class_weight="balanced",
    )
    y_EA = y["EGFR.mutation.status"]
    plot_ROCs(ax[5:], centers, centers_min, X, i, y_EA, lr, "EGFRm")

    return f


def plot_ROCs(ax, centers, centers_min, X, i, y: pd.Series, lr, gene_label):
    """Generate ROC plots using DDMC, unclustered, k-means, and GMM for a particular feature."""
    folds = 7

    # DDMC full
    plotROC(
        ax[0],
        lr,
        centers.values,
        y.values,
        cv_folds=folds,
        title="DDMC—Full data set" + gene_label,
    )

    # DDMC minimal
    plotROC(
        ax[1],
        lr,
        centers_min.values,
        y.values,
        cv_folds=folds,
        title="DDMC—Complete portion" + gene_label,
    )

    # Unclustered
    X_f = X.loc[:, centers.index].T
    X_f.index = np.arange(X_f.shape[0])
    plotROC(ax[2], lr, X_f.values, y.values, cv_folds=folds, title="Unclustered " + gene_label)

    # Run k-means
    d = X.select_dtypes(include=["float64"]).T.reset_index()
    d.rename(columns={"index": "Patient_ID"}, inplace=True)
    d = d.iloc[:, 1:]

    kmeans = KMeans(n_clusters=30).fit(d.T)

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    c_kmeansT = reshapePatients(kmeans.cluster_centers_.T, X.columns[4:])

    # Regress k-means clusters against STK11 status
    plotROC(
        ax[3], lr, c_kmeansT.values, y.values, cv_folds=folds, title="k-means " + gene_label
    )

    # Run GMM
    gmm = DDMC(
        i, n_components=30, SeqWeight=0, distance_method="Binomial", random_state=15
    ).fit(d)

    c_gmmT = reshapePatients(gmm.transform(), X.columns[4:])

    # Regress GMM clusters against STK11 status
    plotROC(ax[4], lr, c_gmmT.values, y.values, cv_folds=folds, title="GMM " + gene_label)


def reshapePatients(centers, patients):
    df = pd.DataFrame(centers)
    df["Patient_ID"] = patients
    df.iloc[:, :-1] = StandardScaler(with_std=False).fit_transform(
        df.iloc[:, :-1]
    )

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    return find_patients_with_NATandTumor(df.copy(), "Patient_ID", conc=True)