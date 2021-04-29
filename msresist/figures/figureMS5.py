"""
This creates Supplemental Figure 5: Predicting EGFRm/ALKf using DDMC clusters.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from ..logistic_regression import plotClusterCoefficients, plotROC
from .common import subplotLabel, getSetup
from .figure2 import plotMotifs, plotDistanceToUpstreamKinase
from .figureM3 import merge_binary_vectors, find_patients_with_NATandTumor
from .figureM4 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (2, 2), multz={0: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Load Clustering Model from Figure 2
    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    # Import Genotype data
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    y = mOI[~mOI["Sample.ID"].str.contains("IR")]

    # Find centers
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.ncl) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = centers.set_index("Patient_ID")

    # Hypothesis Testing
    assert np.all(y['Sample.ID'] == centers.index)
    centers["EGFRm/ALKf"] = merge_binary_vectors(y, "EGFR.mutation.status", "ALK.fusion").values
    pvals = calculate_mannW_pvals(centers, "EGFRm/ALKf", 1, 0)
    pvals = build_pval_matrix(model.ncl, pvals)
    centers["EGFRm/ALKf"] = centers["EGFRm/ALKf"].replace(0, "WT")
    centers["EGFRm/ALKf"] = centers["EGFRm/ALKf"].replace(1, "EGFRm/ALKf")
    plot_clusters_binaryfeatures(centers, "EGFRm/ALKf", ax[0], pvals=pvals)

    # Reshape data (Patients vs NAT and tumor sample per cluster)
    centers = centers.reset_index().set_index("EGFRm/ALKf")
    centers = find_patients_with_NATandTumor(centers.copy(), "Patient_ID", conc=True)
    y = find_patients_with_NATandTumor(y.copy(), "Sample.ID", conc=False)
    assert all(centers.index.values == y.index.values), "Samples don't match"

    # Normalize
    centers = centers.T
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    centers = centers.T

    # Logistic Regression
    centers["EGFRm/ALKf"] = merge_binary_vectors(y, "EGFR.mutation.status", "ALK.fusion").values
    lr = LogisticRegressionCV(Cs=2, cv=12, solver="saga", max_iter=10000, n_jobs=-1, penalty="l1", class_weight="balanced")
    plotROC(ax[1], lr, centers.iloc[:, :-1].values, centers["EGFRm/ALKf"], cv_folds=4, title="ROC EGFRm/ALKf")
    plotClusterCoefficients(ax[2], lr.fit(centers.iloc[:, :-1], centers["EGFRm/ALKf"].values), list(centers.columns[:-1]), title="EGFRm/ALKf")

    return f
