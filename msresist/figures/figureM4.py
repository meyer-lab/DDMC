"""
This creates Figure M4.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal
from .figure3 import plotMotifs, plotUpstreamKinases
from .figureM3 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals
from sklearn.linear_model import LogisticRegressionCV
from ..logistic_regression import plotConfusionMatrix, plotROC
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (2, 3), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Phosphoproteomic aberrations associated with molecular signatures
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Load Clustering Model from Figure 2
    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    # Regression against mutation status of driver genes and clusters
    # Import mutation status of TP53, KRAS, EGFR, and ALK fusion
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    mOI = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = mOI.set_index("Sample.ID")

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    centers = pd.DataFrame(model.transform())
    centers.columns = list(np.arange(model.ncl) + 1)
    centers["Patient_ID"] = X.columns[4:]
    centers.columns = list(np.arange(model.ncl) + 1) + ["Patient_ID"]

    # Logistic Regression
    # lr = LogisticRegressionCV(cv=4, solver="liblinear", n_jobs=-1, penalty="l1", class_weight="balanced")
    lr = LogisticRegressionCV(cv=4, solver="saga", max_iter=10000, n_jobs=-1, penalty="elasticnet", class_weight="balanced", l1_ratios=[0.2, 0.9])

    # TP53 mutation status
    centers["TP53 status"] = y["TP53.mutation.status"].values
    centers = centers.set_index("Patient_ID")
    pvals = calculate_mannW_pvals(centers, "TP53 status", 1, 0)
    pvals = build_pval_matrix(model.ncl, pvals)
    plot_clusters_binaryfeatures(centers, "TP53 status", ax[0], pvals=pvals)
    plotROC(ax[1], lr, centers.iloc[:, :-1].values, centers["TP53 status"], cv_folds=4)

    # plot Cluster Motifs
    pssms = model.pssms(PsP_background=False)
    motifs = [pssms[11], pssms[18]]
    plotMotifs(motifs, titles=["Cluster 12", "Cluster 19"], axes=ax[2:4])

    # plot Upstream Kinases
    plotUpstreamKinases(model, ax=ax[4], clusters_=[12, 19], n_components=2, pX=1)

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
