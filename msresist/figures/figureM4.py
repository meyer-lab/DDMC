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
    ax, f = getSetup((15, 12), (2, 3))

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
    mOI = mutations[["Sample.ID", "TP53.mutation.status", "KRAS.mutation.status", "EGFR.mutation.status", "ALK.fusion"]]
    mOI = mOI[~mOI["Sample.ID"].str.contains("IR")]

    # #### Import clusters
    centers = model.transform()
    y = mOI.set_index("Sample.ID")

    # Logistic Regression
    # EGFR mutation status
    y_egfr = y["EGFR.mutation.status"]

    lr = LogisticRegressionCV(cv=model.ncl, solver="saga", max_iter=10000, n_jobs=-1, penalty="elasticnet", class_weight="balanced", l1_ratios=[0.5, 0.9])

    lr_egfr = lr.fit(centers, y_egfr)

    plotConfusionMatrix(ax[0], lr_egfr, centers, y_egfr)
    ax[0].set_title("EGFR confusion matrix")
    plotROC(ax[1], lr_egfr, centers, y_egfr, cv_folds=4)
    ax[1].set_title("EGFR ROC")
    plotClusterCoefficients(ax[2], lr_egfr)
    ax[2].set_title("EGFR Cluster Coefficients")

    # TP53 mutation status
    y_tp53 = y["TP53.mutation.status"]

    lr_tp53 = lr.fit(centers, y_tp53)

    plotConfusionMatrix(ax[3], lr_tp53, centers, y_tp53)
    ax[3].set_title("TP53 confusion matrix")
    plotROC(ax[4], lr_tp53, centers, y_tp53, cv_folds=4)
    ax[4].set_title("TP53 ROC")
    plotClusterCoefficients(ax[5], lr_tp53)
    ax[5].set_title("TP53 Cluster Coefficients")

    return f
