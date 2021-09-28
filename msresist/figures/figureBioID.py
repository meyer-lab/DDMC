"""
This creates BioID figure
"""

import pandas as pd
from msresist.pca import plotBootPCA, bootPCA, preprocess_ID
import seaborn as sns
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    bid = preprocess_ID(linear=True, npepts=7, FCcut=10)

    # Scores
    bootScor_m, bootScor_sd, bootLoad_m, bootLoad_sd, bootScor, varExp = bootPCA(bid, 4, "Gene", method="NMF", n_boots=100)
    plotBootPCA(ax[0], bootScor_m, bootScor_sd, varExp, title="NMF Scores", LegOut=False, annotate=False, colors=False)
    ax[0].legend(prop={'size': 10})

    plotBootPCA(ax[2], bootScor_m, bootScor_sd, varExp, title="NMF Scores", X="PC2", Y="PC3", LegOut=False, annotate=False, colors=False)
    ax[2].legend(prop={'size': 10})

    # Loadings
    plotBootPCA(ax[1], bootLoad_m, bootLoad_sd, varExp, title="NMF Loadings", LegOut=False, annotate=True, colors=False)
    ax[1].get_legend().remove()

    plotBootPCA(ax[3], bootLoad_m, bootLoad_sd, varExp, title="NMF Loadings", X="PC2", Y="PC3", LegOut=False, annotate=True, colors=False)
    ax[3].get_legend().remove()

    return f
