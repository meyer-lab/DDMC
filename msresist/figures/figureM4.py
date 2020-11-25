"""
This creates Figure M4.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .figure3 import plotR2YQ2Y, plotActualVsPredicted, plotScoresLoadings
from sklearn.linear_model import LogisticRegressionCV
from ..pre_processing import filter_NaNpeptides
from ..logistic_regression import plotClusterCoefficients, plotPredictionProbabilities, plotConfusionMatrix, plotROC
from sklearn.cross_decomposition import PLSRegression
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (3, 3))

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

    # #### Import clusters
    centers = model.transform()
    y = mOI.set_index("Sample.ID")

    # Logistic Regression
    lr = LogisticRegressionCV(cv=4, solver="liblinear", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")

    # TP53 mutation status
    predict_mutants(ax[:3], centers, y, lr, "TP53.mutation.status")
    predict_mutants(ax[3:6], centers, y, lr, "EGFR.mutation.status", mutant2="ALK.fusion")
    predict_mutants(ax[6:9], centers, y, lr, "STK11.mutation.status")

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
