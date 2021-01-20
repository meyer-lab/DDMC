"""
This creates Figure 5.
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from .figureM2 import SwapPatientIDs, AddTumorPerPatient
from .figureM3 import build_pval_matrix, calculate_mannW_pvals, plot_clusters_binaryfeatures
from .figure3 import plotPCA, plotMotifs, plotUpstreamKinases
from .figureM4 import merge_binary_vectors, find_patients_with_NATandTumor


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((17, 10), (3, 3))

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Add subplot labels
    subplotLabel(ax)

    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    centers = pd.DataFrame(model.transform())
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    centers.columns = np.arange(model.ncl) + 1
    centers["Patient_ID"] = X.columns[4:]
    centers = centers.sort_values(by="Patient_ID")

    # Import infiltration data
    y = pd.read_csv("msresist/data/MS/CPTAC/xCellSign_minimal.csv").sort_values(by="Patient ID").dropna(axis=1)
    # y = y.drop("Tregs", axis=1)
    centers = find_patients_with_NATandTumor(centers, "Patient_ID", conc=True)
    y = find_patients_with_NATandTumor(y, "Patient ID", conc=False)
    l1 = list(centers.index)
    l2 = list(y.index)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    centers = centers.drop(dif)

    # Normnalize
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    y.iloc[:, :] = StandardScaler().fit_transform(y.iloc[:, :])

    # Infiltration data PCA
    plotPCA(ax[:2], y.reset_index(), 2, ["Patient ID"], "Cell Line", hue_scores=None, style_scores=None, style_load=None, legendOut=False)

    # LASSO regression
    reg = MultiTaskLassoCV(cv=7, max_iter=10000, tol=0.2).fit(centers, y)
    plot_LassoCoef_Immune(ax[2:4], reg, centers, y, model.ncl)

    # plot Cluster Motifs
    pssms = model.pssms(PsP_background=False)
    motifs = [pssms[5], pssms[8], pssms[19]]
    plotMotifs(motifs, titles=["Cluster 6", "Cluster 9", "Cluster 20"], axes=ax[4:7])

    # plot Upstream Kinases
    plotUpstreamKinases(model, ax=ax[7:9], clusters_=[6, 9, 20], n_components=4, pX=1)

    return f


def plot_LassoCoef_Immune(ax, reg, centers, y, ncl):
    """Plot LASSO coefficients of centers explaining immune infiltration"""
    coef_T = pd.DataFrame(reg.coef_.T).iloc[:24, :]
    coef_T.columns = y.columns
    coef_T["Cluster"] = np.arange(ncl) + 1
    coef_T = pd.melt(coef_T, id_vars="Cluster", value_vars=list(coef_T.columns[:-1]), var_name=["Cell Line"], value_name="Coefficient")
    sns.barplot(x="Cluster", y="Coefficient", hue="Cell Line", data=coef_T, ax=ax[0])
    ax[0].get_legend().remove()
    ax[0].set_title("Tumor Clusters")
    ax[0].set_ylim(-1.5, 2.5)

    coef_NAT = pd.DataFrame(reg.coef_.T).iloc[24:, :]
    coef_NAT.columns = y.columns
    coef_NAT["Cluster"] = np.arange(ncl) + 1
    coef_NAT = pd.melt(coef_NAT, id_vars="Cluster", value_vars=list(coef_NAT.columns[:-1]), var_name=["Cell Line"], value_name="Coefficient")
    sns.barplot(x="Cluster", y="Coefficient", hue="Cell Line", data=coef_NAT, ax=ax[1])
    ax[1].get_legend().remove()
    ax[1].set_title("NAT Clusters")
    ax[1].set_ylim(-1.5, 2.5)

    # Add r2 coef
    textstr = "$r2 score$ = " + str(np.round(r2_score(y, reg.predict(centers)), 4))
    props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
    ax[1].text(0.65, 0.10, textstr, transform=ax[1].transAxes, verticalalignment="top", bbox=props, fontsize=10)
