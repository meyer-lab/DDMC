"""
This creates Figure 6: STK11m downregulates TIICs
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from .common import subplotLabel, getSetup
from ..logistic_regression import plotROC, plotClusterCoefficients
from .figure3 import plotMotifs, plotUpstreamKinase_heatmap


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 12), (2, 3), multz={4: 1})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Add subplot labels
    subplotLabel(ax)

    # Import DDMC clusters
    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.ncl) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = centers.loc[~centers["Patient_ID"].str.endswith(".N"), :].sort_values(by="Patient_ID").set_index("Patient_ID")

    # Import Cold-Hot Tumor data
    y = pd.read_csv("msresist/data/MS/CPTAC/Hot_Cold.csv").dropna(axis=1).sort_values(by="Sample ID")
    y = y.loc[~y["Sample ID"].str.endswith(".N"), :].set_index("Sample ID")
    l1 = list(centers.index)
    l2 = list(y.index)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    centers = centers.drop(dif)

    # Transform to binary
    y = y.replace("Cold-tumor enriched", 0)
    y = y.replace("Hot-tumor enriched", 1)
    y = np.squeeze(y)

    # Remove NAT-enriched samples
    centers = centers.drop(y[y == "NAT enriched"].index)
    y = y.drop(y[y == "NAT enriched"].index).astype(int)
    assert all(centers.index.values == y.index.values), "Samples don't match"

    # Select clusters changed by STK status (from figure M4)
    coi = [1, 5, 7, 9, 11, 12, 15, 19, 21, 22, 24]
    centers = centers.loc[:, coi]

    # Normalize
    centers = centers.T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    centers = centers.T

    # Logistic Regression
    lr = LogisticRegressionCV(cv=5, solver="saga", max_iter=100000, tol=1e-4, n_jobs=-1, penalty="l1", class_weight="balanced")
    plotROC(ax[0], lr, centers.values, y, cv_folds=4, title="ROC TIIC")
    plotClusterCoefficients(ax[1], lr.fit(centers, y.values), xlabels=coi, title="TIIC")

    # Motifs
    pssms = model.pssms(PsP_background=False)
    motifs = [pssms[18], pssms[23]]
    plotMotifs(motifs, titles=["Cluster 19", "Cluster 24"], axes=ax[2:4])

    # plot Upstream Kinases
    plotUpstreamKinase_heatmap(model, [11, 18, 19, 21, 24], ax[4])

    return f
