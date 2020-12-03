"""
This creates Supplemental Figure 1.
"""

import pickle
import numpy as np
from ..figures.figure3 import plotMotifs, plotCenters
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((8, 20), (5, 2))

    with open('msresist/data/pickled_models/AXLmodel_PAM250_W2_5CL', 'rb') as p:
        model = pickle.load(p)[0]

    pssms = model.pssms(PsP_background=False)
    all_lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]

    for ii, jj in zip(range(0, 10, 2), range(5)):
        cluster = "Cluster " + str(jj + 1)
        plotMotifs([pssms[jj]], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], axes=[ax[ii]], titles=[cluster], yaxis=[-35, 12.5])
        plotCenters(ax[ii + 1], model.transform()[:, jj], all_lines, title=cluster, yaxis=[-1, 1])

    # Add subplot labels
    subplotLabel(ax)

    return f
