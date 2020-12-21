"""
This creates Supplemental Figure 4.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import KMeans
from pomegranate import GeneralMixtureModel, NormalDistribution
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from ..logistic_regression import plotClusterCoefficients, plotConfusionMatrix, plotROC
from .figureM3 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 17), (5, 2), multz={4:1, 8:1})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Add subplot labels
    subplotLabel(ax)

    # TP53 unclustered
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, cut=1)
    d = X.set_index("Gene").select_dtypes(include=["float64"]).T.reset_index()
    d.rename(columns={"index": "Patient_ID"},  inplace=True)

    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    mOI = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = mOI.set_index("Sample.ID")
    d["TP53 status"] = y["TP53.mutation.status"].values

    lr = LogisticRegressionCV(cv=4, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")
    uc_lr = lr.fit(d, y)
    plotROC(ax[0], uc_lr, d.values, y, cv_folds=4, title="ROC unclustered")
    plot_unclustered_LRcoef(ax[1], uc_lr, d)