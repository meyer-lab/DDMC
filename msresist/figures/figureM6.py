"""
This creates Figure 6: Tumor infiltrating immune cells
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from .common import subplotLabel, getSetup
from .figureM4 import build_pval_matrix, calculate_mannW_pvals, plot_clusters_binaryfeatures, plot_GO
from .figure2 import plotPCA, plotDistanceToUpstreamKinase
from ..logistic_regression import plotROC, plotClusterCoefficients


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((17, 10), (3, 4), multz={0: 1})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Add subplot labels
    subplotLabel(ax)

    # Import DDMC clusters
    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.ncl) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = centers.loc[~centers["Patient_ID"].str.endswith(".N"), :].sort_values(by="Patient_ID").set_index("Patient_ID")

    ### -------- Build a model to predict Cold- or Hot- tumor enriched samples  -------- ###
    # Import Cold-Hot Tumor data
    cent1, y = FormatXYmatrices(centers.copy())

    # Normalize
    cent1 = cent1.T
    cent1.iloc[:, :] = StandardScaler(with_std=False).fit_transform(cent1.iloc[:, :])
    cent1 = cent1.T

    # Hypothesis Testing
    cent1["TIIC"] = y.values
    pvals = calculate_mannW_pvals(cent1, "TIIC", 1, 0)
    pvals = build_pval_matrix(model.ncl, pvals)
    cent1["TIIC"] = cent1["TIIC"].replace(0, "CTE")
    cent1["TIIC"] = cent1["TIIC"].replace(1, "HTE")
    plot_clusters_binaryfeatures(cent1, "TIIC", ax[0], pvals=pvals, loc="lower left")

    # Logistic Regression
    lr = LogisticRegressionCV(cv=5, solver="saga", max_iter=100000, tol=1e-10, n_jobs=-1, penalty="l2", class_weight="balanced")
    plotROC(ax[1], lr, cent1.iloc[:, :-1].values, y, cv_folds=4, title="ROC TIIC")
    plotClusterCoefficients(ax[2], lr.fit(cent1.iloc[:, :-1], y.values), title="TIIC weights")

    # plot Upstream Kinases
    plotDistanceToUpstreamKinase(model, [2, 6], ax[3], num_hits=2)

    # GO
    plot_GO(2, ax[4], n=4, title="GO Cluster 2")
    plot_GO(6, ax[5], n=4, title="GO Cluster 6")

    ### -------- Build a model that predicts STK, build a model that predicts infiltration, and look for shared cluster dependencies -------- ###
    centers, y = FormatXYmatrices(centers)
    coi = [21, 24] # Top 2 lowest p-values of clusters changed by STK11 status
    centers = centers.loc[:, coi]

    # Normalize
    centers = centers.T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    centers = centers.T

    # Logistic Regression
    lr = LogisticRegressionCV(cv=5, solver="saga", max_iter=100000, tol=1e-10, n_jobs=-1, penalty="l2", class_weight="balanced")
    plotROC(ax[6], lr, centers.values, y, cv_folds=4, title="ROC STIK11-TIIC")
    plotClusterCoefficients(ax[7], lr.fit(centers, y.values), xlabels=coi, title="STK11-TIIC weights")

    # plot Upstream Kinases
    plotDistanceToUpstreamKinase(model, [21, 24], ax[8], num_hits=1, title="STK11-TIIC kinases")

    # GO
    plot_GO(21, ax[9], n=4, title="GO Cluster 21")
    plot_GO(24, ax[10], n=4, title="GO Cluster 24")

    return f


def FormatXYmatrices(centers):
    """Make sure Y matrix has the same matching samples has the signaling centers"""
    y = pd.read_csv("msresist/data/MS/CPTAC/Hot_Cold.csv").dropna(axis=1).sort_values(by="Sample ID")
    y = y.loc[~y["Sample ID"].str.endswith(".N"), :].set_index("Sample ID")
    l1 = list(centers.index)
    l2 = list(y.index)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    centers = centers.drop(dif)

    # Transform to binary
    y = y.replace("Cold-tumor enriched", 0)
    y = y.replace("Hot-tumor enriched", 1)
    y = np.squeeze(y)

    # Remove NAT-enriched samples
    centers = centers.drop(y[y == "NAT enriched"].index)
    y = y.drop(y[y == "NAT enriched"].index).astype(int)
    assert all(centers.index.values == y.index.values), "Samples don't match"
    return centers, y
