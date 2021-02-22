"""
This creates Supplemental Figure 6: Predictive performance of DDMC clusters using different weights
"""

import glob
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from ..logistic_regression import plotROC
from .figureM4 import find_patients_with_NATandTumor, merge_binary_vectors


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 15), (5, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Signaling
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    X = filter_NaNpeptides(X, cut=1)

    # Genotype data
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)

    # Load pickled models with varying weigths
    models = glob.glob('msresist/data/pickled_models/binomial/*')

    # LASSO
    lr = LogisticRegressionCV(Cs=10, cv=10, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")

    folds = 7
    for idx, PathToModel in enumerate(models):
        with open(PathToModel, 'rb') as m:
            model = pickle.load(m)[0]

        print(model.SeqWeight)
        # Find and scale centers
        centers = TransformCenters(model, X)

        #STK11
        plotROC(ax[idx], lr, centers.values, y["STK11.mutation.status"], cv_folds=folds, title="STK " + "w=" + str(model.SeqWeight))

        #EGFRm/ALKf
        y_EA = merge_binary_vectors(y.copy(), "EGFR.mutation.status", "ALK.fusion")
        plotROC(ax[idx + 1], lr, centers.values, y_EA, cv_folds=folds, title="EGFRm/ALKf " + "w=" + str(model.SeqWeight))

        #Hot-Cold behavior
        y_hcb = HotColdBehavior(centers)
        plotROC(ax[idx + 2], lr, centers.values, y_hcb, cv_folds=folds, title="Infiltration " + "w=" + str(model.SeqWeight))

    return f


def TransformCenters(model, X):
    """For a given model, find centers and transform for regression."""
    centers = pd.DataFrame(model.transform()).T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    centers = centers.T
    centers.columns = np.arange(model.ncl) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = find_patients_with_NATandTumor(centers.copy(), "Patient_ID", conc=True)
    return centers


def HotColdBehavior(centers):
    # Import Cold-Hot Tumor data
    centers = centers.loc[~centers["Patient_ID"].str.endswith(".N"), :].sort_values(by="Patient_ID").set_index("Patient_ID")
    y = pd.read_csv("msresist/data/MS/CPTAC/Hot_Cold.csv").dropna(axis=1).sort_values(by="Sample ID")
    y = y.loc[~y["Sample ID"].str.endswith(".N"), :].set_index("Sample ID")
    l1 = list(centers.index)
    l2 = list(y.index)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    centers = y.drop(dif)

    # Transform to binary
    y = y.replace("Cold-tumor enriched", 0)
    y = y.replace("Hot-tumor enriched", 1)
    y = np.squeeze(y)

    # Remove NAT-enriched samples
    centers = centers.drop(y[y == "NAT enriched"].index)
    y = y.drop(y[y == "NAT enriched"].index).astype(int)

    return y