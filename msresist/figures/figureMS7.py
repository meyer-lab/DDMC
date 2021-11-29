"""
This creates Supplemental Figure 7: Predicting STK11 genotype using different clustering strategies.
"""

import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse.construct import random
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from msresist.clustering import MassSpecClustering
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from ..logistic_regression import plotROC
from .figureM4 import find_patients_with_NATandTumor


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (2, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Signaling
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], cut=1)

    # Fit DDMC to complete data
    d = np.array(X.select_dtypes(include=['float64']).T)
    i = X.select_dtypes(include=['object'])

    assert np.all(np.isfinite(d))
    model_min = MassSpecClustering(i, ncl=30, SeqWeight=100, distance_method="Binomial").fit(d)

    centers_min = pd.DataFrame(model_min.transform()).T
    centers_min.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers_min.iloc[:, :])
    centers_min = centers_min.T
    centers_min.columns = np.arange(model_min.n_components) + 1
    centers_min["Patient_ID"] = X.columns[4:]
    centers_min = find_patients_with_NATandTumor(centers_min.copy(), "Patient_ID", conc=True)

    # Fit DDMC
    model = MassSpecClustering(i, ncl=30, SeqWeight=100, distance_method="Binomial").fit(d)

    # Find and scale centers
    centers = pd.DataFrame(model.transform()).T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    centers = centers.T
    centers.columns = np.arange(model.n_components) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = find_patients_with_NATandTumor(centers.copy(), "Patient_ID", conc=True)

    # Predicting STK11
    lr = LogisticRegressionCV(cv=5, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced", random_state=10)
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)
    assert all(centers.index.values == y.index.values), "Samples don't match"
    y_STK = y["STK11.mutation.status"]
    plot_ROCs(ax[:5], centers, centers_min, X, i, y_STK, lr, "STK11")

    # Predicting EGFRm
    lr = LogisticRegressionCV(cv=20, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")
    y_EA = y["EGFR.mutation.status"]
    plot_ROCs(ax[5:], centers, centers_min, X, i, y_EA, lr, "EGFRm")

    return f


def plot_ROCs(ax, centers, centers_min, X, i, y, lr, gene_label):
    """Generate ROC plots using DDMC, unclustered, k-means, and GMM for a particular feature."""
    folds = 7

    # DDMC full
    plotROC(ax[0], lr, centers.values, y, cv_folds=folds, title="DDMC—Full data set" + gene_label)

    # DDMC minimal
    plotROC(ax[1], lr, centers_min.values, y, cv_folds=folds, title="DDMC—Complete portion" + gene_label)

    # Unclustered
    X_f = X.loc[:, centers.index].T
    X_f.index = np.arange(X_f.shape[0])
    plotROC(ax[2], lr, X_f.values, y, cv_folds=folds, title="Unclustered " + gene_label)

    # Run k-means
    ncl = 30
    d = X.select_dtypes(include=["float64"]).T.reset_index()
    d.rename(columns={"index": "Patient_ID"}, inplace=True)
    d = d.iloc[:, 1:]
    x_ = X.copy()
    x_["Cluster"] = KMeans(n_clusters=ncl).fit(d.T).labels_
    c_kmeans = x_.groupby("Cluster").mean().T
    c_kmeans["Patient_ID"] = X.columns[4:]
    c_kmeans.columns = list(np.arange(ncl) + 1) + ["Patient_ID"]
    c_kmeans.iloc[:, :-1] = StandardScaler(with_std=False).fit_transform(c_kmeans.iloc[:, :-1])

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    c_kmeansT = find_patients_with_NATandTumor(c_kmeans.copy(), "Patient_ID", conc=True)

    # Regress k-means clusters against STK11 status
    plotROC(ax[3], lr, c_kmeansT.values, y, cv_folds=folds, title="k-means " + gene_label)

    # Run GMM
    gmm = MassSpecClustering(i, ncl=30, SeqWeight=0, distance_method="Binomial", random_state=15).fit(d, "NA")
    x_["Cluster"] = gmm.labels()
    c_gmm = x_.groupby("Cluster").mean().T
    c_gmm["Patient_ID"] = X.columns[4:]
    c_gmm.iloc[:, :-1] = StandardScaler(with_std=False).fit_transform(c_gmm.iloc[:, :-1])

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    c_gmmT = find_patients_with_NATandTumor(c_gmm.copy(), "Patient_ID", conc=True)

    # Regress GMM clusters against STK11 status
    plotROC(ax[4], lr, c_gmmT.values, y, cv_folds=folds, title="GMM " + gene_label)
