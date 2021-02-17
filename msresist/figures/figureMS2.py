"""
This creates Supplemental Figure 2: Cluster motifs
"""

import pickle
from .common import subplotLabel, getSetup
from ..figures.figure3 import plotMotifs


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 20), (6, 4))

    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    pssms = model.pssms(PsP_background=False)
    for ii in range(model.ncl):
        cluster = "Cluster " + str(ii + 1)
        plotMotifs([pssms[ii]], axes=[ax[ii]], titles=[cluster], yaxis=False)

    # Add subplot labels
    subplotLabel(ax)

    return f
