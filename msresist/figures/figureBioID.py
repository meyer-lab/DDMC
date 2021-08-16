import pandas as pd
from msresist.pca import plotBootPCA, bootPCA, preprocess_ID
import seaborn as sns 
from .common import subplotLabel, getSetup

def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 5), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    bid = preprocess_ID(linear=True)

    #Scores
    bootScor_m, bootScor_sd, bootLoad_m, bootLoad_sd, bootScor = bootPCA(bid, 3, "Gene", method="NMF", n_boots=100)
    plotBootPCA(ax[0], bootScor_m, bootScor_sd, "NMF Scores", LegOut=False, annotate=False, colors=False)
    ax[0].legend(prop={'size': 10})

    #Loadings
    plotBootPCA(ax[1], bootLoad_m, bootLoad_sd, "NMF Loadings", LegOut=False, annotate=True, colors=False)
    ax[1].get_legend().remove()

    return f

