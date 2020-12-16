"""
This creates Figure 5.
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from ..figures.figureM2 import SwapPatientIDs, AddTumorPerPatient
from ..figures.figureM3 import build_pval_matrix, calculate_mannW_pvals, plot_clusters_binaryfeatures
from ..figures.figure3 import plotPCA, plotMotifs, plotUpstreamKinases


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((25, 10), (2, 5), multz={2: 2})

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
    y = y[~y["Patient ID"].str.endswith(".N")]
    y = y.drop("Tregs", axis=1)
    li1 = list(y["Patient ID"])
    li2 = list(centers["Patient_ID"])
    dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    centers = centers.set_index("Patient_ID").drop(dif).reset_index()
    assert all(centers["Patient_ID"].values == y["Patient ID"].values), "sampels not matching"

    # Normnalize Y matrix
    y.iloc[:, 1:] = StandardScaler().fit_transform(y.iloc[:, 1:])

    # Infiltration data PCA
    plotPCA(ax[:2], y, 2, ["Patient ID"], "Cell Line", hue_scores=None, style_scores=None, style_load=None, legendOut=False)

    # LASSO regression
    y = y.set_index("Patient ID")
    centers = centers.set_index("Patient_ID")
    reg = MultiTaskLassoCV(cv=7).fit(centers, y)
    plot_LassoCoef_Immune(ax[2], reg, centers, y, model.ncl)

    # plot Cluster Motifs
    pssms = model.pssms(PsP_background=False)
    motifs = [pssms[5], pssms[8], pssms[19]]
    plotMotifs(motifs, titles=["Cluster 6", "Cluster 9", "Cluster 20"], axes=ax[3:6])

    # plot Upstream Kinases
    plotUpstreamKinases(model, ax=ax[6:8], clusters_=[6, 9, 20], n_components=4, pX=1)

    return f


def plot_LassoCoef_Immune(ax, reg, centers, y, ncl):
    """Plot LASSO coefficients of centers explaining immune infiltration"""
    coef = pd.DataFrame(reg.coef_.T)
    coef.columns = y.columns
    coef["Cluster"] = np.arange(ncl) + 1
    m = pd.melt(coef, id_vars="Cluster", value_vars=list(coef.columns[:-1]), var_name=["Cell Line"], value_name="Coefficient")
    sns.barplot(x="Cluster", y="Coefficient", hue="Cell Line", data=m, ax=ax)
    ax.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0, labelspacing=0.2)
    ax.set_title("Clusters driving Immune Infiltration Signatures")

    # Add r2 coef
    textstr = "$r2 score$ = " + str(np.round(r2_score(y, reg.predict(centers)), 4))
    props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
    ax.text(0.85, 0.10, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props)
