"""
This creates Supplemental Figure 3: Specific phosphosites.
"""

import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from ..pre_processing import preprocessing
from .figure1 import plot_IdSites, plot_AllSites
from ..pca import plotPCA_scoresORloadings
from ..motifs import MapMotifs


all_lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"]


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 4), (1, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Read in Mass Spec data
    X = preprocessing(AXLm_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_col=True)

    # PCA
    data = X.set_index(["Gene"]).select_dtypes(include=float)
    data.columns = all_lines
    plotPCA_scoresORloadings(ax[0], data.reset_index(), 3, ["Gene"], "Signaling", plot="loadings")

    # Specific p-sites
    erk = {"MAPK1": "Y187-p", "MAPK3": "Y204-p"}
    erk_rn = ["ERK2", "ERK1"]

    plot_AllSites(ax[1], X.copy(), "AXL", "AXL", ylim=[-3, 2.5])
    ax[1].legend(loc='upper left', prop={'size': 8})
    plot_AllSites(ax[2], X.copy(), "EGFR", "EGFR", ylim=[-3, 2.5])
    plot_IdSites(ax[3], X.copy(), erk, "ERK1/2", rn=erk_rn, ylim=[-3, 2.5])

    return f
