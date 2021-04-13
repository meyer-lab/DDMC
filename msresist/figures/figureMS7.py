"""
This creates Supplemental Figure 8: Cluster motifs of MCF7 cells ((Hijazi et al Nat Biotech 2020))
"""

import pickle
from .common import subplotLabel, getSetup
from .figure2 import plotMotifs


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 4))

    with open('msresist/data/pickled_models/ebdt_mcf7_pam250_CL12_W5', 'rb') as m:
        model = pickle.load(m)[0]

    pssms = model.pssms(PsP_background=False)
    for ii in range(model.ncl):
        cluster = "Cluster " + str(ii + 1)
        plotMotifs([pssms[ii]], axes=[ax[ii]], titles=[cluster], yaxis=[-50, 20])

    # Add subplot labels
    subplotLabel(ax)

    return f