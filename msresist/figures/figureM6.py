"""
This creates Figure 6: Benchmarking upstream kinases from EBDT (Hijazi et al Nat Biotech 2020) in breast cancer
"""

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from msresist.validations import preprocess_ebdt_mcf7
from .common import subplotLabel, getSetup
from .figure2 import plotPCA, plotMotifs, plotUpstreamKinase_heatmap

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 2), multz={2:1, 4:1})

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    x = preprocess_ebdt_mcf7()
    with open('msresist/data/pickled_models/ebdt_mcf7_binom_CL20_W5', 'rb') as m:
        model = pickle.load(m)

    centers = pd.DataFrame(model.transform())
    centers.columns = np.arange(model.ncl) + 1
    centers.insert(0, "Sample", x.columns[3:])
    centers["Sample"] = [s.split(".")[1].split(".")[0]  for s in centers["Sample"]]

    # PCA
    plotPCA(ax[0:2], centers, 2, ["Sample"], "Cluster")

    # first plot heatmap of clusters
    ax[2].axis("off")

    # Upstream Kinases
    plotUpstreamKinase_heatmap(model, [1, 19], ax[3])

    return f 

