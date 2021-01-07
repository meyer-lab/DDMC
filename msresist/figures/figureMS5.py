"""
This creates Supplemental Figure M4.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from ..logistic_regression import plotClusterCoefficients, plotROC
from .common import subplotLabel, getSetup
from .figure3 import plotMotifs, plotUpstreamKinases
from .figureM3 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals
from .figureM4 import merge_binary_vectors, find_patients_with_NATandTumor


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (2, 4), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Load Clustering Model from Figure 2
    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    # Regression against mutation status of driver genes and clusters
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    centers = pd.DataFrame(model.transform())
    centers.columns = list(np.arange(model.ncl) + 1)
    centers["Patient_ID"] = X.columns[4:]
    centers.columns = list(np.arange(model.ncl) + 1) + ["Patient_ID"]

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    centersT = find_patients_with_NATandTumor(centers.copy(), "Patient_ID", conc=True)
    yT = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)
    assert all(centersT.index.values == yT.index.values), "Samples don't match"

    # Logistic Regression
    lr = LogisticRegressionCV(cv=4, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")
    centers.iloc[:, :-1] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :-1])
    centersT.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centersT.iloc[:, :])

    # EGFR mutation status
    centers["EGFR"] = y["EGFR.mutation.status"].values
    centersT["EGFR"] = yT["EGFR.mutation.status"].values
    centers = centers.set_index("Patient_ID")
    pvals = calculate_mannW_pvals(centers, "EGFR", 1, 0)
    pvals = build_pval_matrix(model.ncl, pvals)
    plot_clusters_binaryfeatures(centers, "EGFR", ax[0], pvals=pvals)
    plotROC(ax[1], lr, centersT.iloc[:, :-1].values, centersT["EGFR"], cv_folds=4, title="ROC EGFR")
    plotClusterCoefficients(ax[2], lr.fit(centersT.iloc[:, :-1], centersT["EGFR"].values), title="EGFR")

    # plot Cluster Motifs
    pssms = model.pssms(PsP_background=False)
    motifs = [pssms[0], pssms[16]]
    plotMotifs(motifs, titles=["Cluster 1", "Cluster 17"], axes=ax[3:5])

    # plot Upstream Kinases
    plotUpstreamKinases(model, ax=ax[5:7], clusters_=[1, 17], n_components=4, pX=1)

    return f
