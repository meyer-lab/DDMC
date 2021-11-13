"""
This creates Supplemental Figure 2: Cluster motifs
"""

import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from .figure2 import plotMotifs
from ..pre_processing import filter_NaNpeptides
from ..clustering import MassSpecClustering


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 9), (5, 5))

    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="white", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Import signaling data
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], tmt=2)
    d = X.select_dtypes(include=[float]).T
    i = X.select_dtypes(include=[object])

    # Fit DDMC
    model = MassSpecClustering(i, ncl=30, SeqWeight=100, distance_method="Binomial", random_state=5).fit(d)

    pssms, cl_num = model.pssms(PsP_background=False)
    ylabels = np.arange(0, 21, 5)
    xlabels = [20, 21, 22, 23, 24, 25]
    for ii, cc in enumerate(cl_num):
        cluster = "Cluster " + str(cc)
        plotMotifs([pssms[ii]], axes=[ax[ii]], titles=[cluster], yaxis=[0, 10])
        if ii not in ylabels:
            ax[ii].set_ylabel("")
            ax[ii].get_yaxis().set_visible(False)
        if ii not in xlabels:
            ax[ii].set_xlabel("")
            ax[ii].get_xaxis().set_visible(False)

    # Add subplot labels
    # subplotLabel(ax) Too many plots to label A-Z

    return f
