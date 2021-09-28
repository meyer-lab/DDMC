"""
This creates Supplemental Figure 6: Predicting EGFRm/ALKf using DDMC clusters.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from msresist.clustering import MassSpecClustering
from sklearn.linear_model import LogisticRegressionCV
from ..logistic_regression import plotClusterCoefficients, plotROC
from .common import subplotLabel, getSetup
from .figure2 import plotMotifs, plotDistanceToUpstreamKinase
from .figureM4 import merge_binary_vectors, find_patients_with_NATandTumor
from .figureM5 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (2, 2), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Load Clustering Model from Figure 2
    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    # Import Genotype data
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)
    assert all(centers.index.values == y.index.values), "Samples don't match"
    y_STK = y["STK11.mutation.status"]
    plot_ROCs(ax[:5], centers, centers_min, X, i, y_STK, "STK11")

    # Predicting EGFRm/Alkf
    y_EA = merge_binary_vectors(y, "EGFR.mutation.status", "ALK.fusion")
    plot_ROCs(ax[5:], centers, centers_min, X, i, y_EA, "EGFRm/ALKf")

    return f


def plot_ROCs(ax, centers, centers_min, X, i, y, gene_label):
    """Generate ROC plots using DDMC, unclustered, k-means, and GMM for a particular feature."""
    # LASSO
    lr = LogisticRegressionCV(cv=5, solver="saga", max_iter=100000, tol=1e-4, n_jobs=-1, penalty="elasticnet", l1_ratios=[0.1])

    folds = 7

    # DDMC full
    plotROC(ax[0], lr, centers.values, y, cv_folds=folds, title="DDMC—Full data set" + gene_label)

    # DDMC minimal
    plotROC(ax[1], lr, centers_min.values, y, cv_folds=folds, title="DDMC—Complete portion" + gene_label)

    # Unclustered
    X_f = X.loc[:, centers.index].T
    X_f.index = np.arange(X_f.shape[0])
    plotROC(ax[2], lr, X_f.values, y, cv_folds=folds, title="Unclustered " + gene_label)

    # Hypothesis Testing
    assert np.all(y['Sample.ID'] == centers.index)
    centers["EGFRm/ALKf"] = merge_binary_vectors(y, "EGFR.mutation.status", "ALK.fusion").values
    pvals = calculate_mannW_pvals(centers, "EGFRm/ALKf", 1, 0)
    pvals = build_pval_matrix(model.ncl, pvals)
    centers["EGFRm/ALKf"] = centers["EGFRm/ALKf"].replace(0, "WT")
    centers["EGFRm/ALKf"] = centers["EGFRm/ALKf"].replace(1, "EGFRm/ALKf")
    plot_clusters_binaryfeatures(centers, "EGFRm/ALKf", ax[0], pvals=pvals)

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    centers = centers.reset_index().set_index("EGFRm/ALKf")
    centers = find_patients_with_NATandTumor(centers.copy(), "Patient_ID", conc=True)
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)
    assert all(centers.index.values == y.index.values), "Samples don't match"

    # Run GMM
    ncl = 15
    gmm = MassSpecClustering(i, ncl=ncl, SeqWeight=0, distance_method="Binomial").fit(d)
    x_["Cluster"] = gmm.labels()
    c_gmm = x_.groupby("Cluster").mean().T
    c_gmm["Patient_ID"] = X.columns[4:]
    c_gmm.iloc[:, :-1] = StandardScaler(with_std=False).fit_transform(c_gmm.iloc[:, :-1])

    # Logistic Regression
    centers["EGFRm/ALKf"] = merge_binary_vectors(y, "EGFR.mutation.status", "ALK.fusion").values
    lr = LogisticRegressionCV(Cs=2, cv=12, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")
    plotROC(ax[1], lr, centers.iloc[:, :-1].values, centers["EGFRm/ALKf"], cv_folds=4, title="ROC EGFRm/ALKf")
    plotClusterCoefficients(ax[2], lr.fit(centers.iloc[:, :-1], centers["EGFRm/ALKf"].values), list(centers.columns[:-1]), title="EGFRm/ALKf")

    return f
