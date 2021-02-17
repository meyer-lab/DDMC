"""
This creates Figure 5.
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import zscore
from .common import subplotLabel, getSetup
from .figureM2 import SwapPatientIDs, AddTumorPerPatient
from .figureM3 import build_pval_matrix, calculate_mannW_pvals, plot_clusters_binaryfeatures
from .figure3 import plotPCA, plotMotifs, plotUpstreamKinase_heatmap
from .figureM4 import find_patients_with_NATandTumor
from ..logistic_regression import plotROC, plotClusterCoefficients


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 12), (3, 3), multz={0: 1, 7: 1})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Add subplot labels
    subplotLabel(ax)

    # Import DDMC clusters 
    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    centers = pd.DataFrame(model.transform()).T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    centers = centers.T
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
    centers = centers.drop(y[y=="NAT enriched"].index)
    y = y.drop(y[y=="NAT enriched"].index).astype(int)
    assert all(centers.index.values == y.index.values), "Samples don't match"

    # Hypothesis Testing
    centers["HCT"] = y.values
    pvals = calculate_mannW_pvals(centers, "HCT", 1, 0)
    pvals = build_pval_matrix(model.ncl, pvals)
    plot_clusters_binaryfeatures(centers, "HCT", ["Cold", "Hot"], ax[0], pvals=pvals)

    # Logistic Regression
    lr = LogisticRegressionCV(Cs=10, cv=4, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")
    plotROC(ax[1], lr, centers.iloc[:, :-1].values, y, cv_folds=4, title="ROC TIIC")
    plotClusterCoefficients(ax[2], lr.fit(centers.iloc[:, :-1], y.values), title="TIIC")

    # Motifs
    pssms = model.pssms(PsP_background=False)
    motifs = [pssms[16], pssms[17], pssms[20]]
    plotMotifs(motifs, titles=["Cluster 17", "Cluster 18", "Cluster 21"], axes=ax[3:6])

    # plot Upstream Kinases
    plotUpstreamKinase_heatmap(model, [17, 18, 20, 21], ax[6])

    return f
