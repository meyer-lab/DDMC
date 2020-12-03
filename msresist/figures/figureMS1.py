"""
This creates Supplemental Figure 1.
"""

import pickle
from .common import subplotLabel, getSetup
from ..figures.figure3 import plotMotifs


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 20), (7, 3))

    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    pssms = model.pssms(PsP_background=False)
    for ii in range(21):
        cluster = "Cluster " + str(ii + 1)
        plotMotifs([pssms[ii]], [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], axes=[ax[ii]], titles=[cluster], yaxis=[-20, 8])

    # Add subplot labels
    subplotLabel(ax)

    return f
