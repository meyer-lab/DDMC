"""
This creates Supplemental Figure 7: Predictive performance of DDMC clusters using binomial vs PAM250
"""


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

    # Load pickled PAM250 and binomial models
    models = []
    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        models.append(pickle.load(p)[0])

    with open('msresist/data/pickled_models/pam250/CPTACmodel_PAM250_CL21_W9_TMT2', 'rb') as p:
        pam250 = pickle.load(p)[0]


    return f