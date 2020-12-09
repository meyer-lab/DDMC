"""
This creates Figure 5.
"""
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from ..figures.figureM2 import SwapPatientIDs, AddTumorPerPatient
from ..figures.figureM3 import build_pval_matrix
from ..figures.figure3 import plotPCA, plotMotifs, plotUpstreamKinases


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((16, 10), (3, 4))

    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    # Import clinical stage data
    cd = pd.read_csv("msresist/data/MS/CPTAC/CPTAC_Clinical_Data_May2019.csv")
    ts = cd[["case_id", "tumor_stage_pathological"]]
    IDict = pd.read_csv("msresist/data/MS/CPTAC/IDs.csv", header=0)
    IDict_ = dict(zip(IDict.iloc[:, 0], IDict.iloc[:, 1]))
    ts = SwapPatientIDs(ts, IDict_).drop("case_id", axis=1)[["Patient_ID", "tumor_stage_pathological"]]
    ts = AddTumorPerPatient(ts).sort_values(by="Patient_ID")

    ts = ts.replace("Stage I", 0)
    ts = ts.replace("Stage IA", 0)
    ts = ts.replace("Stage IB", 1)
    ts = ts.replace("Stage IIA", 2)
    ts = ts.replace("Stage IIB", 3)
    ts = ts.replace("Stage III", 4)
    ts = ts.replace("Stage IIIA", 4)
    ts = ts.replace("Stage IV", 4)

    # Find cluster centers
    centers = pd.DataFrame(model.transform())
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], cut=0.1)
    centers["Patient_ID"] = X.columns[4:]
    centers = centers.sort_values(by="Patient_ID").set_index("Patient_ID").drop(['C3N.02379.1', 'C3N.02587', 'C3N.02587.N']).reset_index()

    #Use tumor samples
    centers = centers[~centers["Patient_ID"].str.endswith(".N")]
    ts = ts[~ts["Patient_ID"].str.endswith(".N")]
    assert list(ts["Patient_ID"]) == list(centers["Patient_ID"]), "Patients don't match"

    # Run Kruskal-Wallis H-test
    centers["Stage"] = np.array(ts.iloc[:, 1])
    centers_ = centers.iloc[:, 1:].set_index("Stage")
    pvals = calculate_Kuskal_pvals(centers_, model)
    data = build_pval_matrix(model.ncl, centers)

    # Plot p-site abundance per stage
    cluster = [8, 12, 14, 16, 19]
    for i, c in enumerate(cluster):
        plot_abundanceVSstage(centers, c, ts, ax=ax[1+i])

    # Motifs and kinase predictions
    pssms = model.pssms(PsP_background=True)
    motifs = [pssms[7], pssms[11], pssms[13], pssms[15], pssms[18]]
    plotMotifs(motifs, [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], titles=["Cluster 8", "Cluster 12", "Cluster 14", "Cluster 16", "Cluster 19"], axes=ax[6:11])
    plotUpstreamKinases(model, ax=ax[11], clusters_=[12, 16], n_components=2, labels=False, pX=1)

    return f


def calculate_Kuskal_pvals(centers, model):
    """Plot Kuskal p-value vs cluster. Note that categorical variables should be converted to numerical."""
    cluster_samples = []
    for ii in range(model.ncl):
        samples = []
        for jj in range(max(centers.index) + 1):
            samples.append(centers.iloc[:, ii].loc[jj].values)
        cluster_samples.append(samples)
    pvals = []
    for c in cluster_samples:
        [*samples] = c
        pval = kruskal(*samples)[1]
        pvals.append(pval)
    return pvals


def plot_abundanceVSstage(centers, cluster, ts, ax):
    """Cluster Abundance in each patient vs Stage"""
    cluster -= 1
    X = centers.loc[:, ["Patient_ID", "Stage", cluster]]
    sns.stripplot(x="Stage", y=cluster, data=X, color="darkblue", ax=ax)
    sns.boxplot(x="Stage", y=cluster, data=X, color="white", linewidth=2, ax=ax)
    ax.set_xticklabels(["IA-", "IB", "IIA", "IIB", "III+"])
    ax.set_title("p-site Abundance")
    ax.set_ylabel("Cluster " + str(cluster + 1))
