"""
This creates Supplemental Figure 4.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from pomegranate import GeneralMixtureModel, NormalDistribution
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from ..logistic_regression import plotClusterCoefficients, plotConfusionMatrix, plotROC
from .figureM3 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals
from .figureM4 import find_patients_with_NATandTumor
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
    d = d.iloc[:, 1:]

    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y_ = y["STK11.mutation.status"]

    lr = LogisticRegressionCV(Cs=10, cv=15, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")

    plotROC(ax[0], lr, d.values, y_, cv_folds=4, title="ROC unclustered")
    plot_unclustered_LRcoef(ax[1], lr.fit(d, y_), d)

    # Run k-means
    ncl = 24
    x_ = X.copy()
    x_["Cluster"] = KMeans(n_clusters=ncl).fit(d.T).labels_
    c_kmeans = x_.groupby("Cluster").mean().T
    c_kmeans["Patient_ID"] = X.columns[4:]
    c_kmeans.columns = list(np.arange(ncl) + 1) + ["Patient_ID"]

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    c_kmeansT = find_patients_with_NATandTumor(c_kmeans.copy(), "Patient_ID", conc=True)
    yT = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)
    assert all(c_kmeansT.index.values == yT.index.values), "Samples don't match"

    # Regress k-means clsuters against STK11 status
    c_kmeans["STK11"] = y["STK11.mutation.status"].values
    c_kmeansT["STK11"] = yT["STK11.mutation.status"].values
    c_kmeans = c_kmeans.set_index("Patient_ID")
    plotROC(ax[2], lr, c_kmeansT.iloc[:, :-1].values, c_kmeansT["STK11"], cv_folds=4, title="ROC k-means")
    plotClusterCoefficients(ax[3], lr.fit(c_kmeansT.iloc[:, :-1], c_kmeansT["STK11"].values), title="k-means")
    pvals = calculate_mannW_pvals(c_kmeans, "STK11", 0, 1)
    pvals = build_pval_matrix(ncl, pvals)
    plot_clusters_binaryfeatures(c_kmeans, "STK11", ax[4], pvals=pvals)

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

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    c_gmmT = find_patients_with_NATandTumor(c_gmm.copy(), "Patient_ID", conc=True)
    assert all(c_gmmT.index.values == yT.index.values), "Samples don't match"

    # Regress GMM clusters against STK11 status
    c_gmm["STK11"] = y["STK11.mutation.status"].values
    c_gmmT["STK11"] = yT["STK11.mutation.status"].values
    c_gmm = c_gmm.set_index("Patient_ID")
    plotROC(ax[5], lr, c_gmmT.iloc[:, :-1].values, c_gmmT["STK11"], cv_folds=4, title="ROC GMM")
    plotClusterCoefficients(ax[6], lr.fit(c_gmmT.iloc[:, :-1], c_gmmT["STK11"].values), title="GMM")
    pvals = calculate_mannW_pvals(c_gmm, "STK11", 0, 1)
    pvals = build_pval_matrix(ncl, pvals)
    plot_clusters_binaryfeatures(c_gmm, "STK11", ax[7], pvals=pvals)

    return f
