"""
This creates Figure M4.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from .figure3 import plotMotifs, plotUpstreamKinases
from .figureM3 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from ..logistic_regression import plotROC, plotClusterCoefficients
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (2, 4), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Phosphoproteomic aberrations associated with molecular signatures
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
    lr = LogisticRegressionCV(cv=4, solver="saga", max_iter=10000, n_jobs=-1, penalty="l2", class_weight="balanced")
    centers.iloc[:, :-1] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :-1])
    centersT.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centersT.iloc[:, :])

    # TP53 mutation status
    centersT["TP53 status"] = yT["TP53.mutation.status"].values
    centers["TP53 status"] = y["TP53.mutation.status"].values
    centers = centers.set_index("Patient_ID")
    pvals = calculate_mannW_pvals(centers, "TP53 status", 1, 0)
    pvals = build_pval_matrix(model.ncl, pvals)
    plot_clusters_binaryfeatures(centers, "TP53 status", ax[0], pvals=pvals)
    plotROC(ax[1], lr, centersT.iloc[:, :-1].values, centersT["TP53 status"], cv_folds=4)
    plotClusterCoefficients(ax[2], lr.fit(centersT.iloc[:, :-1], centersT["TP53 status"].values), title="TP53")

    # plot Cluster Motifs
    pssms = model.pssms(PsP_background=False)
    motifs = [pssms[11], pssms[18]]
    plotMotifs(motifs, titles=["Cluster 12", "Cluster 19"], axes=ax[3:5])

    # plot Upstream Kinases
    plotUpstreamKinases(model, ax=ax[5], clusters_=[12, 19], n_components=2, pX=1)

    return f


def merge_binary_vectors(y, mutant1, mutant2):
    """Merge binary mutation status vectors to identify all patients having one of the two mutations"""
    y1 = y[mutant1]
    y2 = y[mutant2]
    y_ = np.zeros(y.shape[0])
    for binary in [y1, y2]:
        indices = [i for i, x in enumerate(binary) if x == 1]
        y_[indices] = 1
    return pd.Series(y_)


def find_patients_with_NATandTumor(X, label, conc=False):
    """Reshape data to display patients as rows and samples (Tumor and NAT per cluster) as columns.
    Note that to do so, samples that don't have their tumor/NAT counterpart are dropped."""
    xT = X[~X[label].str.endswith(".N")].sort_values(by=label)
    xN = X[X[label].str.endswith(".N")].sort_values(by=label)
    l1 = list(xT[label])
    l2 = [s.split(".N")[0] for s in xN[label]]
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    X = xT.set_index(label).drop(dif)
    assert all(X.index.values == np.array(l2)), "Samples don't match"

    if conc:
        xN = xN.set_index(label)
        xN.index = l2
        xN.columns = [str(i) + "_N" for i in xN.columns]
        X.columns = [str(i) + "_T" for i in X.columns]
        X = pd.concat([X, xN], axis=1)
    return X