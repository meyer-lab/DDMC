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
from ..figures.figure3 import plotPCA, plotMotifs, plotUpstreamKinases


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 10), (3, 2), multz={0: 1})

    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    # Import cancer stage data
    cd = pd.read_csv("msresist/data/MS/CPTAC/CPTAC_Clinical_Data_May2019.csv")
    ts = cd[["case_id", "tumor_stage_pathological"]]
    IDict = pd.read_csv("msresist/data/MS/CPTAC/IDs.csv", header=0)
    IDict_ = dict(zip(IDict.iloc[:, 0], IDict.iloc[:, 1]))
    ts = SwapPatientIDs(ts, IDict_).drop("case_id", axis=1)[["Patient_ID", "tumor_stage_pathological"]]
    ts = AddTumorPerPatient(ts).sort_values(by="Patient_ID")

    ts = ts.replace("Stage I", 0)
    ts = ts.replace("Stage IA", 1)
    ts = ts.replace("Stage IB", 2)
    ts = ts.replace("Stage IIA", 3)
    ts = ts.replace("Stage IIB", 4)
    ts = ts.replace("Stage III", 5)
    ts = ts.replace("Stage IIIA", 6)
    ts = ts.replace("Stage IV", 7)

    # Find cluster centers
    centers = pd.DataFrame(model.transform())
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], cut=0.1)
    centers["Patient_ID"] = X.columns[4:]
    centers = centers.sort_values(by="Patient_ID").set_index("Patient_ID").drop(['C3N.02379.1', 'C3N.02587', 'C3N.02587.N']).reset_index()
    assert list(ts["Patient_ID"]) == list(centers["Patient_ID"]), "Patients don't match"

    # Run Kruskal-Wallis H-test
    plot_ClinicalStaging_Kuskal(centers, ts, model, ax=ax[0])

    # Motifs and kinase predictions
    pssms = model.pssms(PsP_background=False)
    motifs = [pssms[14], pssms[17]]
    plotMotifs(motifs, [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], titles=["Cluster 15", "Cluster 18"], axes=ax[1:3])
    plotUpstreamKinases(model, ax=ax[3:5], clusters_=[15, 18], n_components=4, pX=1)

    return f


def plot_ClinicalStaging_Kuskal(centers, ts, model, ax):
    centers["Stage"] = np.array(ts.iloc[:, 1])
    centers_ = centers.iloc[:, 1:].set_index("Stage")
    cluster_samples = []
    for ii in range(model.ncl):
        samples = []
        for jj in range(max(centers_.index)):
            samples.append(centers_.iloc[:, ii].loc[jj].values)
        cluster_samples.append(samples)
    pvals = []
    for c in cluster_samples:
        s1, s2, s3, s4, s5, s6, s7 = c
        pval = kruskal(s1, s2, s3, s4, s5, s6, s7)[1]
        pvals.append(pval)

    #plot pvalues
    data = pd.DataFrame()
    data["Clusters"] = np.arange(model.ncl)
    data["Kuskal p-value"] = pvals
    signif = []
    for val in pvals:
        if val < 0.05:
            signif.append(True)
        else:
            signif.append(False)
    data["Significant"] = signif
    sns.barplot(x="Clusters", y="Kuskal p-value", hue="Significant", data=data, ax=ax)