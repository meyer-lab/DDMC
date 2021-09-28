"""
This creates Supplemental Figure 3: Predictive performance of DDMC clusters using different weights
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from .common import subplotLabel, getSetup
from ..logistic_regression import plotROC
from .figureM4 import TransformCenters, HotColdBehavior, find_patients_with_NATandTumor, merge_binary_vectors


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (3, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Signaling
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]

    # Genotype data
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)

    # LASSO
    lr = LogisticRegressionCV(Cs=10, cv=10, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")

    folds = 5
    weights = [0, 15, 20, 40, 50]
    path = 'msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W'
    for ii, w in enumerate(weights):
        with open(path + str(w) + '_TMT2', 'rb') as m:
            model = pickle.load(m)[0]

        # Find and scale centers
        centers_gen, centers_hcb = TransformCenters(model, X)

        if w == 0:
            prio = " (data only)"
        elif w == 50:
            prio = " (motif mainly)"
        else:
            prio = " (mix)"

        # STK11
        plotROC(ax[ii], lr, centers_gen.values, y["STK11.mutation.status"], cv_folds=folds, title="STK11m " + "w=" + str(model.SeqWeight) + prio)

        # EGFRm/ALKf
        y_EA = merge_binary_vectors(y.copy(), "EGFR.mutation.status", "ALK.fusion")
        plotROC(ax[ii + 5], lr, centers_gen.values, y_EA, cv_folds=folds, title="EGFRm/ALKf " + "w=" + str(model.SeqWeight) + prio)

        # Hot-Cold behavior
        y_hcb, centers_hcb = HotColdBehavior(centers_hcb)
        plotROC(ax[ii + 10], lr, centers_hcb.values, y_hcb, cv_folds=folds, title="Infiltration " + "w=" + str(model.SeqWeight) + prio)

    return f
