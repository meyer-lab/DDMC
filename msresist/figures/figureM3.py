"""
This creates Figure M3.
"""

import numpy as np
import pandas as pd
import pickle
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegressionCV
from .common import subplotLabel, getSetup
from ..figures.figureM2 import TumorType
from ..logistic_regression import plotClusterCoefficients, plotPredictionProbabilities, plotConfusionMatrix, plotROC
from ..figures.figure3 import plotPCA


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (2, 3))

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]

    with open('CPTACmodel_PAM250_W1_15CL', 'rb') as p:
        model = pickle.load(p)[0]

    centers = pd.DataFrame(model.transform())
    centers["Patient_ID"] = X.columns[4:]
    centers.iloc[:, :-1] = zscore(centers.iloc[:, :-1], axis=1)
    centers.columns = list(np.arange(model.ncl) + 1) + ["Patient_ID"]

    #first plot heatmap of clusters
    ax[0].axis("off")

    #PCA analysis
    centers = TumorType(centers)
    plotPCA(ax[1:3], centers, 2, ["Patient_ID", "Type"], "Cluster", hue_scores="Type", style_scores="Type", hue_load="Cluster")

    #Regression
    c = centers.select_dtypes(include=['float64'])
    tt = centers.iloc[:, -1]
    tt = tt.replace("Normal", 0)
    tt = tt.replace("Tumor", 1)
    lr = LogisticRegressionCV(cv=model.ncl, solver="saga", penalty="l2").fit(c, tt)

    # plotPredictionProbabilities(ax[3], lr, c, tt)
    plotConfusionMatrix(ax[3], lr, c, tt)
    plotROC(ax[4], lr, c, tt, cv_folds=model.ncl)
    plotClusterCoefficients(ax[5], lr)

    # Add subplot labels
    subplotLabel(ax)

    return f
