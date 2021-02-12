"""
This creates Figure 5.
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from .common import subplotLabel, getSetup
from .figureM2 import SwapPatientIDs, AddTumorPerPatient
from .figureM3 import build_pval_matrix, calculate_mannW_pvals
from .figure3 import plotPCA, plotMotifs, plotUpstreamKinase_heatmap
from .figureM4 import find_patients_with_NATandTumor


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 7), (2, 4), multz={2:1, 7: 1})

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Add subplot labels
    subplotLabel(ax)

    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.ncl) + 1
    centers["Patient_ID"] = X.columns[4:]

    # Import infiltration data
    y = pd.read_csv("msresist/data/MS/CPTAC/xCellSign_minimal.csv").sort_values(by="Patient ID").dropna(axis=1)
    centers = find_patients_with_NATandTumor(centers, "Patient_ID", conc=True)
    y = find_patients_with_NATandTumor(y, "Patient ID", conc=False)
    l1 = list(centers.index)
    l2 = list(y.index)
    dif = [i for i in l1 + l2 if i not in l1 or i not in l2]
    centers = centers.drop(dif)
    assert all(centers.index.values == y.index.values), "Samples don't match"

    # Normnalize
    centers.iloc[:, :] = StandardScaler(with_std=False).fit_transform(centers.iloc[:, :])
    y.iloc[:, :] = StandardScaler().fit_transform(y.iloc[:, :])

    # Infiltration data PCA
    plotPCA(ax[:2], y.reset_index(), 2, ["Patient ID"], "Cell Line", hue_scores=None, style_scores=None, style_load=None, legendOut=False)

    # LASSO regression
    reg = MultiTaskLassoCV(cv=10, max_iter=100000, tol=1e-8).fit(centers, y)
    plot_LassoCoef_Immune(ax[2], reg, centers, y, model.ncl, s_type="Tumor")

    # plot Cluster Motifs
    pssms = model.pssms(PsP_background=False)
    motifs = [pssms[1], pssms[2], pssms[16]]
    plotMotifs(motifs, titles=["Cluster 2", "Cluster 3", "Cluster 17"], axes=ax[3:6])

    # plot Upstream Kinases
    plotUpstreamKinase_heatmap(model, [2, 3, 17], ax[6])

    return f


def plot_LassoCoef_Immune(ax, reg, centers, y, ncl, s_type="Tumor"):
    """Plot LASSO coefficients of centers explaining immune infiltration"""
    # Format data for seaborn
    coef = pd.DataFrame(reg.coef_.T)
    coef.columns = y.columns
    coef["Cluster"] = list(np.arange(24)) * 2
    coef["Sample"] = ["Tumor"] * ncl + ["NAT"] * ncl
    coef = pd.melt(coef, id_vars=["Cluster", "Sample"], value_vars=list(coef.columns[:-2]), var_name=["Cell Line"], value_name="Coefficient")

    if s_type:
        coef = coef[coef["Sample"] == s_type]
        ax.set_title(s_type + " samples driving Infiltration")
        sns.barplot(x="Cluster", y="Coefficient", hue="Cell Line", data=coef, ax=ax, **{"linewidth": 0.2}, **{"edgecolor": "black"})
    else:
        ax.set_title("Tumor and NAT samples driving Infiltration")
        sns.catplot(x="Cluster", y="Coefficient", hue="Cell Line", col="Sample", kind="bar", data=coef, ax=ax, **{"linewidth": 0.2}, **{"edgecolor": "black"})

    # Add r2 coef
    textstr = "$r2 score$ = " + str(np.round(r2_score(y, reg.predict(centers)), 4))
    props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
    ax.text(0.65, 0.10, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props, fontsize=10)
