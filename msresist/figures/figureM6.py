"""
This creates Figure 6: Benchmarking upstream kinases from EBDT (Hijazi et al Nat Biotech 2020) in breast cancer
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from ..validations import preprocess_ebdt_mcf7
from .common import subplotLabel, getSetup
from .figure1 import plotPCA_scoresORloadings
from .figure2 import plotPCA, plotDistanceToUpstreamKinase

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 6), (2, 3), multz={3:1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    x = preprocess_ebdt_mcf7()
    with open('msresist/data/pickled_models/ebdt_mcf7_binom_CL20_W5', 'rb') as m:
        model = pickle.load(m)

    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.ncl) + 1
    centers.insert(0, "Inhibitor", x.columns[3:])
    centers["Inhibitor"] = [s.split(".")[1].split(".")[0]  for s in centers["Inhibitor"]]

    # PCA
    AKTi = ["Torin1", "HS173", "GDC0941", "Ku0063794", "AZ20", "MK2206", "AZD5363", "GDC0068", "AZD6738", "AT13148", "Edelfosine", "GF109203X"]
    centers["AKTi"] = [drug in AKTi for drug in centers["Inhibitor"]]
    plotPCA(ax[:2], centers, 2, ["Inhibitor", "AKTi"], "Cluster", hue_scores="AKTi")
    ax[0].legend(loc='lower left', prop={'size': 9}, title="AKTi")

    # Upstream Kinases
    plotDistanceToUpstreamKinase(model, [1], ax[2], num_hits=1)

    # first plot heatmap of clusters
    ax[3].axis("off")

    # Substrates bar plot
    plotSubstratesPerCluster(x, model, "Akt1", ax[4])

    return f 


def plotSubstratesPerCluster(x, model, kinase, ax):
    """Plot normalized number of substrates of a given kinase per cluster."""
    # Refine PsP K-S data set
    ks = pd.read_csv("msresist/data/Validations/Kinase_Substrate_Dataset.csv")
    ks = ks[
    (ks["KINASE"] == "Akt1") & 
    (ks["IN_VIVO_RXN"] == "X") & 
    (ks["IN_VIVO_RXN"] == "X") & 
    (ks["KIN_ORGANISM"] == "human") &
    (ks["SUB_ORGANISM"] == "human")
    ]

    # Count matching substrates per cluster and normalize by cluster size
    x["cluster"] = model.labels()
    counters = {}
    for i in range(1, max(x["cluster"]) + 1):
        counter = 0
        cl = x[x["cluster"] == i]
        put_sub = list(zip(list(cl["gene"]), list(list(cl["pos"]))))
        psp = list(zip(list(ks["SUB_GENE"]), list(ks["SUB_MOD_RSD"])))
        for sub_pos in put_sub:
            if sub_pos in psp:
                counter += 1
        counters[i] = counter / cl.shape[0]

    # Plot
    data = pd.DataFrame()
    data["Cluster"] = counters.keys()
    data["Normalized substrate count"] = counters.values()
    sns.barplot(data=data, x="Cluster", y="Normalized substrate count", color="darkblue", ax=ax, **{"linewidth": 0.5, "edgecolor": "k"})
    ax.set_title(kinase + " Substrate Enrichment")


