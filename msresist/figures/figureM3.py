"""
This creates Figure 2: Validations
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from ..clustering import MassSpecClustering
from ..validations import preprocess_ebdt_mcf7
from .common import subplotLabel, getSetup
from ..pca import plotPCA
from .figure2 import plotDistanceToUpstreamKinase, plotMotifs, ShuffleClusters
from .figureM5 import plot_NetPhoresScoreByKinGroup
from ..clustering import compute_control_pssm
from ..binomial import AAlist
from ..pre_processing import filter_NaNpeptides


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (3, 3), multz={3: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Import signaling data
    x = preprocess_ebdt_mcf7()
    d = x.select_dtypes(include=[float]).T
    i = x.select_dtypes(include=[object])

    # Fit DDMC and find centers
    model = MassSpecClustering(i, ncl=20, SeqWeight=5, distance_method="Binomial", random_state=10).fit(d)
    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.n_components) + 1
    centers.insert(0, "Inhibitor", x.columns[3:])
    centers["Inhibitor"] = [s.split(".")[1].split(".")[0] for s in centers["Inhibitor"]]

    # PCA AKT
    AKTi = ["GSK690693", "Torin1", "HS173", "GDC0941", "Ku0063794", "AZ20", "MK2206", "AZD5363", "GDC0068", "AZD6738", "AT13148", "Edelfosine", "GF109203X", "AZD8055"]
    centers["AKTi"] = [drug in AKTi for drug in centers["Inhibitor"]]
    plotPCA(ax[:2], centers, 2, ["Inhibitor", "AKTi"], "Cluster", hue_scores="AKTi")
    ax[0].legend(loc='lower left', prop={'size': 9}, title="AKTi", fontsize=9)

    # Upstream Kinases AKT EBDT
    plotDistanceToUpstreamKinase(model, [16], ax=ax[2], num_hits=1)

    # first plot heatmap of clusters
    ax[3].axis("off")

    # AKT substrates bar plot
    plot_NetPhoresScoreByKinGroup("msresist/data/cluster_analysis/MCF7_NKIN_CL16.csv", ax[4], title="Cluster 16—Kinase Predictions", n=40)

    # # ERK2 White lab motif
    erk2 = pd.read_csv("msresist/data/Validations/Computational/ERK2_substrates.csv")
    erk2 = compute_control_pssm([s.upper() for s in erk2["Peptide"]])
    erk2 = pd.DataFrame(np.clip(erk2, a_min=0, a_max=3))
    erk2.index = AAlist
    plotMotifs([erk2], axes=[ax[5]], titles=["ERK2"])

    # ERK2 prediction
    # Import signaling data
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=2)
    d = X.select_dtypes(include=[float]).T
    i = X.select_dtypes(include=[object])

    # Fit DDMC
    model_cptac = MassSpecClustering(i, ncl=30, SeqWeight=100, distance_method="Binomial", random_state=5).fit(d)

    s_pssms = ShuffleClusters([3, 7, 21], model_cptac, additional=erk2)
    plotDistanceToUpstreamKinase(model_cptac, [3, 7, 21], additional_pssms=s_pssms + [erk2], add_labels=["3_S", "7_S", "21_S", "ERK2+_S", "ERK2+"], ax=ax[-2:], num_hits=1)

    return f


def plotMCF7AKTclustermap(model, cluster):
    """Code to create hierarchical clustering of cluster 1 across treatments"""
    c1 = pd.DataFrame(model.transform()[:, cluster - 1])
    X = pd.read_csv("msresist/data/Validations/Computational/ebdt_mcf7.csv")
    index = [col.split("7.")[1].split(".")[0] for col in X.columns[2:]]
    c1["Inhibitor"] = index
    c1 = c1.set_index("Inhibitor")
    lim = np.max(np.abs(c1)) * 0.3
    g = sns.clustermap(c1, method="centroid", cmap="bwr", robust=True, vmax=lim, vmin=-lim, row_cluster=True, col_cluster=False, figsize=(2, 15), yticklabels=True, xticklabels=False)
