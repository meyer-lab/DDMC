"""
This creates Supplemental Figure M4.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegressionCV
from ..logistic_regression import plotClusterCoefficients, plotROC
from .common import subplotLabel, getSetup
from .figureM3 import plot_clusters_binaryfeatures, build_pval_matrix, calculate_mannW_pvals
from .figureM4 import merge_binary_vectors
from .figureM5 import plot_abundance_byBinaryFeature


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((17, 10), (3, 5), multz={2:1, 7:1})

    # Add subplot labels
    subplotLabel(ax)

    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Load Clustering Model from Figure 2
    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    # Regression against mutation status of driver genes and clusters
    # Import mutation status of TP53, KRAS, EGFR, and ALK fusion
    mutations = pd.read_csv("msresist/data/MS/CPTAC/Patient_Mutations.csv")
    mOI = mutations[["Sample.ID"] + list(mutations.columns)[45:54] + list(mutations.columns)[61:64]]
    mOI = mOI[~mOI["Sample.ID"].str.contains("IR")]
    y = mOI.set_index("Sample.ID")

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    centers = pd.DataFrame(model.transform())
    centers.columns = list(np.arange(model.ncl) + 1)
    centers["Patient_ID"] = X.columns[4:]
    # centers = centers[~centers["Patient_ID"].str.endswith(".N")] #only tumor samples
    # y = y[~y.index.str.endswith(".N")]

    # Logistic Regression
    lr = LogisticRegressionCV(cv=4, solver="saga", max_iter=10000, n_jobs=-1, penalty="elasticnet", class_weight="balanced", l1_ratios=[0.2, 0.9])
    centers.iloc[:, :-2] = zscore(centers.iloc[:, :-2], axis=0)

    # TP53 MW p-values and LR coefficients #TODO hue lines instead of coloring (hue order?)
    centers["TP53 status"] = y["TP53.mutation.status"].values
    pvals = calculate_mannW_pvals(centers, "TP53 status", 1, 0)
    pvals = build_pval_matrix(model.ncl, pvals)
    pvals["p-value"] = -np.log10(pvals["p-value"])
    plotClusterCoefficients(ax[0], lr.fit(centers.iloc[:, :-2], centers["TP53 status"].values), title="TP53")
    sns.barplot(x="p-value", y="Clusters", data=pvals, orient="h", hue="Significant", ax=ax[1])
    ax[1].set_ylabel("-log10(p-value)")
    ax[1].set_title("Mann-Whitney Test TP53")
    centers = centers.drop("TP53 status", axis=1)

    # EGFRmut + ALKfus
    centers["EGFRm/ALKf"] = merge_binary_vectors(y, "EGFR.mutation.status", "ALK.fusion").iloc[centers.index]
    pvals = calculate_mannW_pvals(centers, "EGFRm/ALKf", 1, 0)
    pvals = build_pval_matrix(model.ncl, pvals)
    plot_clusters_binaryfeatures(centers, "EGFRm/ALKf", ax[2], pvals=pvals)
    plotROC(ax[3], lr, centers.iloc[:, :-2].values, centers["EGFRm/ALKf"], cv_folds=4, title="ROC EGFRm/ALKf")
    plotClusterCoefficients(ax[4], lr.fit(centers.iloc[:, :-2], centers["EGFRm/ALKf"].values), title="EGFRm/ALKf")
    pvals["p-value"] = -np.log10(pvals["p-value"])
    sns.barplot(x="p-value", y="Clusters", data=pvals, orient="h", hue="Significant", ax=ax[5])
    ax[5].set_ylabel("-log10(p-value)")
    ax[5].set_title("Mann-Whitney Test EGFRm/ALKf")
    centers = centers.drop("EGFRm/ALKf", axis=1)

    # STK11
    centers["STK11"] = y["STK11.mutation.status"].values
    pvals = calculate_mannW_pvals(centers, "STK11", 1, 0)
    pvals = build_pval_matrix(model.ncl, pvals)
    plot_clusters_binaryfeatures(centers, "STK11", ax[6], pvals=pvals)
    plotROC(ax[7], lr, centers.iloc[:, :-2].values, centers["STK11"], cv_folds=4, title="ROC STK11")
    plotClusterCoefficients(ax[8], lr.fit(centers.iloc[:, :-2], centers["STK11"].values), title="STK11")
    pvals["p-value"] = -np.log10(pvals["p-value"])
    sns.barplot(x="p-value", y="Clusters", data=pvals, orient="h", hue="Significant", ax=ax[9])
    ax[9].set_ylabel("-log10(p-value)")
    ax[9].set_title("Mann-Whitney Test STK11")

    return f
