"""
This creates KRAS figure clustering only mesenchymal cell lines separately
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from .common import subplotLabel, getSetup
from msresist.pre_processing import MeanCenter
from msresist.validations import pos_to_motif
from msresist.clustering import MassSpecClustering
from msresist.pca import plotPCA
from msresist.figures.figure2 import plotDistanceToUpstreamKinase

sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (5, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Import KRAS signaling data
    ms = pd.read_csv("msresist/data/MS/KRAS_G12C_Haura.csv")

    # Calu Clustering
    calu = add_motifs(ms.iloc[:, :6]).dropna()
    dc = calu.select_dtypes(include=[float]).T
    ic = calu.select_dtypes(include=[object])

    with open("msresist/data/pickled_models/kras/KRAS_Haura_Calu_Binomial_CL15_W30", 'rb') as m:
        calu_model = pickle.load(m)
    calu_centers = centers(calu_model, dc, scale=False)

    plotPCA(ax, calu_centers.reset_index(), 2, ["Time point"], "Cluster", hue_scores="Time point", style_scores="Time point")
    for i in range(2):
        ax[i].axhline(y=0, color="0.25", linestyle="--")
        ax[i].axvline(x=0, color="0.25", linestyle="--")

    cOI = [[2, 5, 15], [1, 4, 7, 10], [11, 14], [3], [6]]
    plot_cluster_centers_timepoints(ax, calu, calu_model, calu_centers, cOI)
    plotDistanceToUpstreamKinase(calu_model, [1, 4, 7, 10, 11, 14], ax, num_hits=4)

    # H1792 Clustering
    with open("msresist/data/pickled_models/kras/KRAS_Haura_H1792_Binomial_CL15_W100", 'rb') as m:
        h1792_model = pickle.load(m)
    h1792 = add_motifs(ms[list(ms.columns[:3]) + list(ms.columns[6:9])]).dropna()
    dh = h1792.select_dtypes(include=[float]).T
    ih = h1792.select_dtypes(include=[object])

    h1792_centers = centers(h1792_model, dh, scale=False)

    plotPCA(ax, h1792_centers.reset_index(), 2, ["Time point"], "Cluster", hue_scores="Time point", style_scores="Time point")
    for i in range(2):
        ax[i].axhline(y=0, color="0.25", linestyle="--")
        ax[i].axvline(x=0, color="0.25", linestyle="--")

    cOI = [[8, 15], [1, 5, 12], [2, 10, 14], [9], [6]]
    plot_cluster_centers_timepoints(ax, h1792, h1792_model, h1792_centers, cOI)
    plotDistanceToUpstreamKinase(h1792_model, [2, 6, 8, 9, 12, 15], ax, num_hits=4)

    return f


def add_motifs(X):
    """Add sequence motifs to KRAS data set."""
    X = MeanCenter(X, mc_row=True, mc_col=False)
    X.insert(1, "Position", [(aa + pos).split(";")[0] for aa, pos in zip(X["Amino Acid"], X["Positions Within Proteins"])])
    X = X.drop(["Amino Acid", "Positions Within Proteins"], axis=1)
    motifs, del_ids = pos_to_motif(X["Gene"], X["Position"])
    X = X.set_index(["Gene", "Position"]).drop(del_ids).reset_index()
    X.insert(0, "Sequence", motifs)
    return X


def centers(model, d, scale=True):
    """Compute cluster centers across cell line and time point."""
    centers = pd.DataFrame(model.transform()).T
    if scale:
        centers = pd.DataFrame(StandardScaler().fit_transform(centers))
    centers.columns = d.index
    centers.index = np.arange(model.ncl) + 1

    cols = centers.columns
    centers = centers.T
    centers["Cell Line"] = [i.split("_")[0] for i in cols]
    centers["Time point"] = [i.split("_")[1] for i in cols]
    return centers


def plot_cluster_centers_timepoints(ax, X, model, centers, cOI):
    """Plotting function across time points."""
    cData = pd.melt(frame=centers, id_vars=["Cell Line", "Time point"], value_vars=centers.columns[:-2], value_name="Center", var_name="Cluster")
    X["Cluster"] = model.labels()
    for axI, lc in enumerate(cOI):
        dd = []
        for c in lc:
            cc = cData[cData["Cluster"] == c]
            cs = X[X["Cluster"] == c]
            cc["Cluster"] = [str(r) + " (n=" + str(cs.shape[0]) + ")" for r in cc["Cluster"]]
            dd.append(cc)
        cD = pd.concat(dd)
        sns.lineplot(data=cD, x="Time point", y="Center", style="Cluster", ax=ax[axI], color=["blue", "red", "green", "orange", "brown"][axI])
        ax[axI].set_title("Center Cluster ")
