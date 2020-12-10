"""
This creates Figure 5.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from ..figures.figureM2 import SwapPatientIDs, AddTumorPerPatient
from ..figures.figureM3 import build_pval_matrix, calculate_mannW_pvals, plot_clusters_binaryfeatures
from ..figures.figure3 import plotPCA, plotMotifs, plotUpstreamKinases


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 20), (7, 4), multz={0:3})

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
    imf = pd.read_csv('msresist/data/MS/CPTAC/cold_hot.csv').iloc[:, :-1].sort_values(by="Sample ID")
    li1 = list(imf["Sample ID"])
    li2 = list(centers["Patient_ID"])
    dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
s.set_index("Patient_ID").drop(dif).reset_index()
    assert all(centers["Patient_ID"].values == imf["Sample ID"].values), "sampels not matching"

    centers["Infiltration"] = imf["Group"].values
    centers = centers.replace("NAT enriched", 0)
    centers = centers.replace("Cold-tumor enriched", 1)
    centers = centers.replace("Hot-tumor enriched", 2)
    centers_MW = centers[~centers["Patient_ID"].str.endswith(".N")]
    centers_MW = centers_MW.set_index("Patient_ID").drop(["C3L.00412", "C3L.02508", "C3N.00738"]).reset_index()

    # Two-way; Cold vs Hot tumors
    pvals = calculate_mannW_pvals(centers_MW, "Infiltration", 0, 1)
    pvals = build_pval_matrix(model.ncl, pvals)
    plot_clusters_binaryfeatures(centers_MW, "Infiltration", ax[0], pvals=pvals, labels=["Cold", "Hot"])

    #Three-way: NAT vs Cold vs Hot tumors
    pvals = calculate_Kuskal_pvals(centers.iloc[:, 1:].set_index("Infiltration"))
    for ii in range(model.ncl):
        plot_abundance_byBinaryFeature(centers.reset_index(), ii+1, "Infiltration", ["NAT", "Cold", "Hot"], ax[1 + ii])

    return f


def calculate_Kuskal_pvals(centers):
    """Plot Kuskal p-value vs cluster. Note that categorical variables should be converted to numerical."""
    ncl = max(centers.columns)
    cluster_samples = []
    for ii in range(ncl):
        samples = []
        for jj in range(max(centers.index) + 1):
            samples.append(centers.iloc[:, ii].loc[jj].values)
        cluster_samples.append(samples)
    pvals = []
    for c in cluster_samples:
        [*samples] = c
        pval = kruskal(*samples)[1]
        pvals.append(pval)
    pvals = multipletests(pvals)[1] #p-value correction for multiple tests
    return pvals


def plot_abundance_byBinaryFeature(centers, cluster, feature, xlabels, ax):
    """Plot infiltration status per cluster"""
    X = centers.loc[:, [feature, cluster]]
    sns.stripplot(x=feature, y=cluster, data=X, color="darkblue", ax=ax)
    sns.boxplot(x=feature, y=cluster, data=X, color="white", linewidth=2, ax=ax)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("p-site Abundance")
    ax.set_title("Cluster " + str(cluster))


