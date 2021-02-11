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
from .figure3 import plotPCA, plotMotifs, plotUpstreamKinase_heatmap
from .figureM4 import find_patients_with_NATandTumor


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((20, 15), (3, 3), multz={7: 1})

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
    reg = MultiTaskLassoCV(cv=10, max_iter=1000000, n_jobs=-1).fit(centers, y)
    plot_LassoCoef_Immune(ax[2:4], reg, centers, y, model.ncl)

    # plot Cluster Motifs
    pssms = model.pssms(PsP_background=False)
    motifs = [pssms[1], pssms[2], pssms[16]]
    plotMotifs(motifs, titles=["Cluster 2", "Cluster 3", "Cluster 17"], axes=ax[4:7])

    # plot Upstream Kinases
    plotUpstreamKinase_heatmap(model, [2, 3, 17], ax[7])

    return f


def plot_LassoCoef_Immune(ax, reg, centers, y, ncl):
    """Plot LASSO coefficients of centers explaining immune infiltration"""
    # Format data for seaborn
    coef = pd.DataFrame(reg.coef_.T)
    coef.columns = y.columns
    coef["Cluster"] = list(np.arange(24)) * 2
    coef["Sample"] = ["Tumor"] * ncl + ["NAT"] * ncl
    coef = pd.melt(coef, id_vars=["Cluster", "Sample"], value_vars=list(coef.columns[:-2]), var_name=["Cell Line"], value_name="Coefficient")

    # Plot tumor
    coef_T = coef[coef["Sample"] == "Tumor"]
    sns.barplot(x="Cluster", y="Coefficient", hue="Cell Line", data=coef_T, ax=ax[0], **{"linewidth": 0.2}, **{"edgecolor": "black"})
    ax[0].get_legend().remove()
    ax[0].set(ylim=(-1.5, 2.5))
    ax[0].set_title("Tumor")

    # Plot NAT
    coef_NAT = coef[coef["Sample"] == "NAT"]
    sns.barplot(x="Cluster", y="Coefficient", hue="Cell Line", data=coef_NAT, ax=ax[1], **{"linewidth": 0.2}, **{"edgecolor": "black"})
    ax[1].set(ylim=(-1.5, 2.5))
    ax[1].set_title("NAT")
    ax[1].legend(labelspacing=0.1)
    plt.setp(ax[1].get_legend().get_texts(), fontsize='10')
    plt.setp(ax[1].get_legend().get_title(), fontsize='9')

    # Add r2 coef
    textstr = "$r2 score$ = " + str(np.round(r2_score(y, reg.predict(centers)), 4))
    props = dict(boxstyle="square", facecolor="none", alpha=0.5, edgecolor="black")
    ax[1].text(0.65, 0.10, textstr, transform=ax[1].transAxes, verticalalignment="top", bbox=props, fontsize=10)
