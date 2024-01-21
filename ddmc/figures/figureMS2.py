"""
This creates Supplemental Figure 2: Cluster motifs
"""

import numpy as np
from .common import getSetup, getDDMC_CPTAC
from .common import plotMotifs


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 9), (5, 5))

    # Fit DDMC
    model, _ = getDDMC_CPTAC(n_components=30, SeqWeight=100.0)

    pssms, cl_num = model.pssms(PsP_background=False)
    ylabels = np.arange(0, 21, 5)
    xlabels = [20, 21, 22, 23, 24, 25]
    for ii, cc in enumerate(cl_num):
        cluster = "Cluster " + str(cc)
        plotMotifs(pssms[ii], ax=ax[ii], titles=cluster, yaxis=[0, 10])
        if ii not in ylabels:
            ax[ii].set_ylabel("")
            ax[ii].get_yaxis().set_visible(False)
        if ii not in xlabels:
            ax[ii].set_xlabel("")
            ax[ii].get_xaxis().set_visible(False)

    return f
