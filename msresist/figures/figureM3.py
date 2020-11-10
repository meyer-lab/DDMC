"""
This creates Figure M3.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegressionCV
from .common import subplotLabel, getSetup
from ..figures.figureM2 import TumorType
from ..logistic_regression import plotClusterCoefficients, plotPredictionProbabilities, plotConfusionMatrix, plotROC
from ..figures.figure3 import plotPCA, plotMotifs, plotUpstreamKinase_heatmap
from ..clustering import MassSpecClustering
from ..pre_processing import filter_NaNpeptides, MeanCenter
import pickle


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 20), (4, 3), multz={9:2})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]

    with open('msresist/data/pickled_models/CPTACmodel_BINOMIAL_CL24_W100_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    centers = pd.DataFrame(model.transform())
    centers["Patient_ID"] = X.columns[4:]
    centers.iloc[:, :-1] = zscore(centers.iloc[:, :-1], axis=1)
    centers.columns = list(np.arange(model.ncl) + 1) + ["Patient_ID"]

    # first plot heatmap of clusters
    ax[0].axis("off")

    # PCA analysis
    centers = TumorType(centers)
    plotPCA(ax[1:3], centers, 2, ["Patient_ID", "Type"], "Cluster", hue_scores="Type", style_scores="Type", hue_load="Cluster")

    # Regression
    c = centers.select_dtypes(include=['float64'])
    tt = centers.iloc[:, -1]
    tt = tt.replace("Normal", 0)
    tt = tt.replace("Tumor", 1)
    lr = LogisticRegressionCV(cv=model.ncl, solver="saga", max_iter=10000, n_jobs=-1, penalty="elasticnet", class_weight="balanced", l1_ratios=[0.5, 0.9]).fit(c, tt)

    # plotPredictionProbabilities(ax[3], lr, c, tt)
    plotConfusionMatrix(ax[3], lr, c, tt)
    plotROC(ax[4], lr, c.values, tt, cv_folds=4)
    plotClusterCoefficients(ax[5], lr)

    # plot Cluster Motifs 
    pssms = model.pssms(PsP_background=True)
    motifs = [pssms[7], pssms[19], pssms[22]]
    plotMotifs(motifs, [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], titles=["Cluster 8", "Cluster 20", "Cluster 23"], ax=ax[6:9])

    # plot Upstream Kinases
    plotUpstreamKinase_heatmap(model, [8, 20, 23], ax=ax[9])

    # Add subplot labels
    subplotLabel(ax)

    return f
