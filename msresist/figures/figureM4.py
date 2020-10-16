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
    ax, f = getSetup((10, 10), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # # Phosphoproteomic aberrations associated with molecular signatures
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # ## Load Clustering Model from Figure 2
    ncl = 15
    with open('CPTACmodel_PAM250_W1_15CL', 'rb') as ff:
        MSC = pickle.load(ff)[0]

    # ## Regression against mutation status of driver genes and clusters
    # 
    # #### Import mutation status of TP53, KRAS, EGFR, and ALK fusion 
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID", "TP53.mutation.status", "KRAS.mutation.status", "EGFR.mutation.status", "ALK.fusion"]]
    mOI = mOI[~mOI["Sample.ID"].str.contains("IR")]

    # #### Import clusters
    centers = MSC.transform()
    y = mOI.set_index("Sample.ID")

    # ### Logistic Regression
    # 
    # EGFR mutation status
    y_egfr = y["EGFR.mutation.status"]

    lr = LogisticRegressionCV(cv=ncl, solver="saga", max_iter=10000, n_jobs=-1, penalty="elasticnet", class_weight="balanced", l1_ratios=[0.5, 0.9])

    lr.fit(centers, y_egfr)
    y_pred = lr.predict(centers)

    plotPredictionProbabilities(ax[1], lr, y_pred, centers, y_egfr)
    plotConfusionMatrix(ax[2], lr, centers, y_egfr)
    plotROC(ax[3], lr, centers, y_egfr, cv_folds=ncl)
    plotClusterCoefficients(ax[4], lr)

    # TP53 mutation status
    y_tp53 = y["TP53.mutation.status"]

    lr.fit(centers, y_tp53)
    y_pred = lr.predict(centers)

    plotPredictionProbabilities(ax[5], lr, y_pred, centers, y_tp53)
    plotConfusionMatrix(ax[6], lr, centers, y_tp53)
    plotROC(ax[7], lr, centers, y_tp53, cv_folds=ncl)
    plotClusterCoefficients(ax[8], lr)

    return f
