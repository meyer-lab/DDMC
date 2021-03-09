"""
This creates Figure 6: Benchmarking upstream kinases from EBDT (Hijazi et al Nat Biotech 2020) in breast cancer
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from msresist.validations import preprocess_ebdt_mcf7
from msresist.clustering import MassSpecClustering
from .common import subplotLabel, getSetup
from .figure3 import plotPCA, plotMotifs, plotUpstreamKinase_heatmap

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (3, 3), multz={0:1, 7: 1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    x = preprocess_ebdt_mcf7()
    with open('msresist/data/pickled_models/ebdt_mcf7_pam250_CL12_W5', 'rb') as m:
        pam_model = pickle.load(m)[0]

    centers = pd.DataFrame(pam_model.transform())
    centers.columns = np.arange(pam_model.ncl) + 1
    centers.insert(0, "Sample", x.columns[3:])
    centers["Sample"] = [s.split(".")[1].split(".")[0]  for s in centers["Sample"]]

    # first plot heatmap of clusters
    ax[0].axis("off")

    # PCA
    plotPCA(ax[1:3], centers, 2, ["Sample"], "Cluster")

    # Motifs
    pssms = pam_model.pssms(PsP_background=False)
    motifs = [pssms[1], pssms[5], pssms[10]]
    plotMotifs(motifs, titles=["Cluster 2", "Cluster 6", "Cluster 11"], axes=ax[3:6])

    # Upstream Kinases
    plotUpstreamKinase_heatmap(pam_model, list(np.arange(pam_model.ncl) + 1), ax[6])

    return f 

