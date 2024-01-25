"""
This creates Supplemental Figure 3: Predictive performance of DDMC clusters using different weights
"""

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from .common import getSetup, getDDMC_CPTAC
from .figureM4 import TransformCenters, HotColdBehavior, find_patients_with_NATandTumor
from ..logistic_regression import plotROC


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (3, 5))

    # Genotype data
    mutations = pd.read_csv("ddmc/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[
        ["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]
    ]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)

    # LASSO
    lr = LogisticRegressionCV(
        Cs=10,
        cv=10,
        solver="saga",
        max_iter=10000,
        n_jobs=-1,
        penalty="l1",
        class_weight="balanced",
    )

    folds = 5
    weights = [0, 100, 500, 1000, 1000000]
    for ii, w in enumerate(weights):
        model, X = getDDMC_CPTAC(n_components=30, SeqWeight=w)

        # Find and scale centers
        centers_gen, centers_hcb = TransformCenters(model, X)

        if w == 0:
            prio = " (data only)"
        elif w == 50:
            prio = " (motif mainly)"
        else:
            prio = " (mix)"

        # STK11
        plotROC(
            ax[ii],
            lr,
            centers_gen.values,
            y["STK11.mutation.status"].values, # type: ignore
            cv_folds=folds,
            title="STK11m " + "w=" + str(model.seq_weight) + prio,
        )

        # EGFRm
        plotROC(
            ax[ii + 5],
            lr,
            centers_gen.values,
            y["EGFR.mutation.status"].values, # type: ignore
            cv_folds=folds,
            title="EGFRm " + "w=" + str(model.seq_weight) + prio,
        )

        # Hot-Cold behavior
        y_hcb, centers_hcb = HotColdBehavior(centers_hcb)
        plotROC(
            ax[ii + 10],
            lr,
            centers_hcb.values,
            y_hcb.values,
            cv_folds=folds,
            title="Infiltration " + "w=" + str(model.seq_weight) + prio,
        )

    return f
