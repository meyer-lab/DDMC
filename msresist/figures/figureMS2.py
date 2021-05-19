"""
This creates Supplemental Figure 2: Cluster motifs
"""

import pickle
import numpy as np 
from .common import subplotLabel, getSetup
from .figure2 import plotMotifs


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((7, 10), (6, 4))

    with open('msresist/data/pickled_models/binomial/CPTACmodel_BINOMIAL_CL24_W15_TMT2', 'rb') as p:
        model = pickle.load(p)[0]

    pssms = model.pssms(PsP_background=False)
    ylabels = np.arange(0, 21, 4)
    xlabels = [20, 21, 22, 23, 24]
    for ii in range(model.ncl):
        cluster = "Cluster " + str(ii + 1)
        plotMotifs([pssms[ii]], axes=[ax[ii]], titles=[cluster], yaxis=[0, 10])
        if ii not in ylabels:
            ax[ii].set_ylabel("")
            ax[ii].get_yaxis().set_visible(False)
        elif ii not in xlabels:
            ax[ii].set_xlabel("")
            ax[ii].get_xaxis().set_visible(False)

    # Add subplot labels
    subplotLabel(ax)

    return f
