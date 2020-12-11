"""
This creates Figure M4.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .figure3 import plotMotifs, plotUpstreamKinases
from .figureM3 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals
from sklearn.linear_model import LogisticRegressionCV
from ..pre_processing import filter_NaNpeptides
from ..logistic_regression import plotClusterCoefficients, plotPredictionProbabilities, plotConfusionMatrix, plotROC
from sklearn.cross_decomposition import PLSRegression
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
    centers["Patient_ID"] = X.columns[4:]
    centers.columns = list(np.arange(model.ncl) + 1) + ["Patient_ID"]
    cc = centers[~centers["Patient_ID"].str.endswith(".N")]  # only tumor samples
    yy = y[~y.index.str.endswith(".N")]

    # Logistic Regression
    lr1 = LogisticRegressionCV(cv=4, solver="liblinear", n_jobs=-1, penalty="l1", class_weight="balanced")
    # lr2 = LogisticRegressionCV(cv=4, solver="saga", max_iter=10000, n_jobs=-1, penalty="elasticnet", class_weight="balanced", l1_ratios=[0.2, 0.9])

    # TP53 mutation status
    cc["TP53 status"] = yy["TP53.mutation.status"].values
    pvals = build_pval_matrix(model.ncl, cc, "TP53 status", 1, 0)
    plot_clusters_binaryfeatures(cc, "TP53 status", ax[0], pvals=pvals)
    plotROC(ax[1], lr1, model.transform(), y["TP53.mutation.status"], cv_folds=4)

    # plot Cluster Motifs
    pssms = model.pssms(PsP_background=False)
    motifs = [pssms[11], pssms[18]]
    plotMotifs(motifs, [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], titles=["Cluster 12", "Cluster 19"], axes=ax[2:4])

    # plot Upstream Kinases
    plotUpstreamKinases(model, ax=ax[4], clusters_=[12, 19], n_components=2, pX=1)

    # EGFRmut + ALKfus
    # predict_mutants(ax[5:8], model.transform(), y, lr1, "EGFR.mutation.status", mutant2="ALK.fusion")
    # cc["EGFRm/ALKf"] = merge_binary_vectors(y, "EGFR.mutation.status", "ALK.fusion").iloc[cc.index]
    # plot_clusters_binaryfeatures(cc, "EGFRm/ALKf", ax[8])
    # pvals = build_pval_matrix(calculate_mannW_pvals(cc, "EGFRm/ALKf", 1, 0), model.ncl)
    # sns.barplot(x="p-value", y="Clusters", data=pvals, hue="Significant", orient="h", ax=ax[9])
    # cc = cc.drop("EGFRm/ALKf", axis=1)

    # STK11
    # predict_mutants(ax[10:13], model.transform(), y, lr2, "STK11.mutation.status")
    # cc["STK11"] = yy["STK11.mutation.status"].values
    # plot_clusters_binaryfeatures(cc, "STK11", ax[13])
    # pvals = build_pval_matrix(calculate_mannW_pvals(cc, "STK11", 1, 0), model.ncl)
    # sns.barplot(x="p-value", y="Clusters", data=pvals, hue="Significant", orient="h", ax=ax[14])

    return f


def predict_mutants(ax, centers, y, lr, mutant, mutant2=False):
    """Determine what mutants to predict"""
    if mutant2:
        y_ = merge_binary_vectors(y, mutant, mutant2)
    else:
        y_ = y[mutant]

    lr_ = lr.fit(centers, y_)
    plotConfusionMatrix(ax[0], lr_, centers, y_)
    ax[0].set_title(mutant.split(".")[0] + "m Confusion Matrix")
    if mutant2:
        ax[0].set_title(mutant.split(".")[0] + "m/" + mutant2.split(".")[0] + " Confusion Matrix")
    plotROC(ax[1], lr_, centers, y_, cv_folds=4)
    if mutant2:
        ax[1].set_title(mutant.split(".")[0] + "m/" + mutant2.split(".")[0] + " Cluster Coefficients")
    ax[1].set_title(mutant.split(".")[0] + "m ROC")
    plotClusterCoefficients(ax[2], lr_)
    if mutant2:
        ax[2].set_title(mutant.split(".")[0] + "m/" + mutant2.split(".")[0] + " Cluster Coefficients")
    else:
        ax[2].set_title(mutant.split(".")[0] + "m Cluster Coefficients")


def merge_binary_vectors(y, mutant1, mutant2):
    """Merge binary mutation status vectors to identify all patients having one of the two mutations"""
    y1 = y[mutant1]
    y2 = y[mutant2]
    y_ = np.zeros(y.shape[0])
    for binary in [y1, y2]:
        indices = [i for i, x in enumerate(binary) if x == 1]
        y_[indices] = 1
    return pd.Series(y_)
