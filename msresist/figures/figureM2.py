"""
This creates Figure M2.
"""

import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.image as mpimg
from sklearn.linear_model import LogisticRegressionCV
from .common import subplotLabel, getSetup
from ..clustering import MassSpecClustering
from ..figures.figure1 import plotClustergram
from ..figures.figureM1 import TumorType
from ..pre_processing import filter_NaNpeptides
from ..figures.figure3 import plotPCA
from ..logistic_regression import plotClusterCoefficients, plotPredictionProbabilities, plotConfusionMatrix, plotROC


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (2, 3))

    # Import MS data and filter peptides with excessive missingness
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X_f = filter_NaNpeptides(X, cut=0.1)
    X_f.index = np.arange(X_f.shape[0])
    d_f = X_f.select_dtypes(include=['float64']).T
    i_f = X_f.select_dtypes(include=['object'])

    # Run model
    ncl = 15
    MSC = MassSpecClustering(i_f, ncl=ncl, SeqWeight=1, distance_method="PAM250").fit(d_f, "NA")

    with open('CPTACmodel_PAM250_W1_15CL', 'wb') as f:
        pickle.dump([MSC], f)

    centers = MSC.transform()
    centers["Patient_ID"] = X.columns[4:]
    centers.iloc[:, :-1] = zscore(centers.iloc[:, :-1], axis=1)
    centers.columns = list(np.arange(ncl) + 1) + ["Patient_ID"]

    # # Clustergram of model centers––sns.clustermap needs the entire plot, inserting image for now
    # img = mpimg.imread('../../doc/_static/stinkbug.png')
    # ax[0].imshow(img)

    # #PCA Analysis
    # centers = TumorType(centers)
    # plotPCA(ax[1], centers, 2, ["Patient_ID", "Type"], "Cluster", hue_scores="Type", style_scores="Type", hue_load="Cluster")

    # # Regression against sample type (Tumor vs NAT)
    # c = centers.iloc[:, 1:-1]
    # c.iloc[:, 1:] = zscore(c.iloc[:, 1:])
    # tt = centers.iloc[:, -1]
    # tt = tt.replace("Normal", 0)
    # tt = tt.replace("Tumor", 1)
    # lr = LogisticRegressionCV(cv=ncl, solver="saga", penalty="l1").fit(c, tt)
    # y_pred = lr.predict(c)

    # plotPredictionProbabilities(ax[2], lr, y_pred, c, tt)
    # plotConfusionMatrix(ax[3], lr, c, tt)
    # plotROC(ax[4], lr, c, tt, cv_folds=ncl)
    # plotClusterCoefficients(ax[5], lr)

    # # Add subplot labels
    # subplotLabel(ax)

    return f
