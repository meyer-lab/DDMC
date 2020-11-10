"""
This creates Figure 5.
"""
import pickle
import numpy as np
import pandas as pd
from ..pre_processing import filter_NaNpeptides
from ..figures.figureM2 import SwapPatientIDs, AddTumorPerPatient
from .common import subplotLabel, getSetup
from ..logistic_regression import plotROC, plotConfusionMatrix, plotClusterCoefficients
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (2, 3))

    with open('msresist/data/pickled_models/CPTACmodel_BINOMIAL_CL24_W100_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    # Import cancer stage data
    cd = pd.read_csv("msresist/data/MS/CPTAC/CPTAC_Clinical_Data_May2019.csv")
    ts = cd[["case_id", "tumor_stage_pathological"]]
    IDict = pd.read_csv("msresist/data/MS/CPTAC/IDs.csv", header=0)
    IDict_ = dict(zip(IDict.iloc[:, 0], IDict.iloc[:, 1]))
    ts = SwapPatientIDs(ts, IDict_).drop("case_id", axis=1)[["Patient_ID", "tumor_stage_pathological"]]
    ts = AddTumorPerPatient(ts).sort_values(by="Patient_ID")
    ts = ts.replace("Stage I", 0)
    ts = ts.replace("Stage IA", 0)
    ts = ts.replace("Stage IB", 0)
    ts = ts.replace("Stage IIA", 0)
    ts = ts.replace("Stage IIB", 1)
    ts = ts.replace("Stage III", 1)
    ts = ts.replace("Stage IIIA", 1)
    ts = ts.replace("Stage IV", 1)

    # ts = ts.replace("Stage I", 0)
    # ts = ts.replace("Stage IA", 1)
    # ts = ts.replace("Stage IB", 2)
    # ts = ts.replace("Stage IIA", 3)
    # ts = ts.replace("Stage IIB", 4)
    # ts = ts.replace("Stage III", 5)
    # ts = ts.replace("Stage IIIA", 6)
    # ts = ts.replace("Stage IV", 7)

    # Find cluster centers
    centers = pd.DataFrame(model.transform())
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], cut=0.1)
    centers["Patient_ID"] = X.columns[4:]
    centers = centers.sort_values(by="Patient_ID").set_index("Patient_ID").drop(['C3N.02379.1', 'C3N.02587', 'C3N.02587.N']).reset_index()
    assert list(ts["Patient_ID"]) == list(centers["Patient_ID"]), "Patients don't match"

    x, y = np.array(centers.iloc[:, 1:]), np.array(ts.iloc[:, 1])
    y = pd.DataFrame(y)

    # Run Logistic Regression
    lr = LogisticRegressionCV(cv=model.ncl, solver="saga", max_iter=10000, n_jobs=-1, penalty="elasticnet", class_weight="balanced", l1_ratios=[0.5, 0.7]).fit(x, y)
    plotConfusionMatrix(ax[0], lr, x, y)
    ax[0].set_title("LR Confusion Matrix")
    plotROC(ax[1], lr, x, y, cv_folds=4)
    ax[1].set_title("LR ROC")
    plotClusterCoefficients(ax[2], lr)

    # Run SVC
    clf = LinearSVC(penalty="l1", dual=False, class_weight="balanced").fit(x, y)
    plotConfusionMatrix(ax[3], clf, x, y)
    ax[3].set_title("SVC Confusion Matrix")
    plotROC(ax[4], clf, x, pd.DataFrame(y), cv_folds=4)
    ax[4].set_title("SVC ROC")
    plotClusterCoefficients(ax[5], clf)
    ax[5].set_title("SVC Cluster Coefficients")

    return f