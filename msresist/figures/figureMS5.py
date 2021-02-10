"""
This creates Supplemental Figure 5: Predicting STK11 genotype using different clustering strategies.
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pomegranate import GeneralMixtureModel, NormalDistribution
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from ..logistic_regression import plotClusterCoefficients, plotConfusionMatrix, plotROC
from .figureM3 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals
from .figureM4 import find_patients_with_NATandTumor, merge_binary_vectors
from .figureMS3 import plot_unclustered_LRcoef


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (2, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Signaling
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, cut=1)

    # Load Clustering Model from Figure 2
    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    # Find and scale centers
    centers = pd.DataFrame(model.transform())
    centers["Patient_ID"] = X.columns[4:]
    centers.columns = list(np.arange(model.ncl) + 1) + ["Patient_ID"]
    centers.iloc[:, :-1] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :-1])
    centers = find_patients_with_NATandTumor(centers.copy(), "Patient_ID", conc=True)

    # Predicting STK11
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)
    assert all(centers.index.values == y.index.values), "Samples don't match"
    y_STK = y["STK11.mutation.status"]
    plot_ROCs(ax[:4], centers, X, y_STK, "STK11")

    # Predicting EGFRm/Alkf
    y_EA = merge_binary_vectors(y, "EGFR.mutation.status", "ALK.fusion")
    plot_ROCs(ax[4:8], centers, X, y_EA, "EGFRm/ALKf")

    return f


def plot_ROCs(ax, centers, X, y, gene_label):
    """Generate ROC plots using DDMC, unclustered, k-means, and GMM for a particular feature."""
    # LASSO
    lr = LogisticRegressionCV(Cs=10, cv=15, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")

    # DDMC
    plotROC(ax[0], lr, centers.values, y, cv_folds=4, title="DDMC " + gene_label)

    # Unclustered
    X_f = X.loc[:, centers.index].T
    X_f.index = np.arange(X_f.shape[0])
    plotROC(ax[1], lr, X_f.values, y, cv_folds=4, title="Unclustered " + gene_label)

    # Run k-means
    ncl = 24
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
    plotROC(ax[2], lr, c_kmeansT.values, y, cv_folds=4, title="k-means " + gene_label)

    # Run GMM
    ncl = 15
    for _ in range(10):
        gmm = GeneralMixtureModel.from_samples(NormalDistribution, X=d.T, n_components=ncl, n_jobs=-1)
        scores = gmm.predict_proba(d.T)
        if np.all(np.isfinite(scores)):
            break
    x_["Cluster"] = gmm.predict(d.T)
    c_gmm = x_.groupby("Cluster").mean().T
    c_gmm["Patient_ID"] = X.columns[4:]
    c_gmm.columns = list(np.arange(ncl) + 1) + ["Patient_ID"]
    c_gmm.iloc[:, :-1] = StandardScaler(with_std=False).fit_transform(c_gmm.iloc[:, :-1])

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    c_gmmT = find_patients_with_NATandTumor(c_gmm.copy(), "Patient_ID", conc=True)

    # Regress GMM clusters against STK11 status
    plotROC(ax[3], lr, c_gmmT.values, y, cv_folds=4, title="GMM " + gene_label)
