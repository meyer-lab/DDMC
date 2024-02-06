"""
This creates Supplemental Figure 2: Cluster motifs
"""

import numpy as np

from ddmc.clustering import DDMC
from ddmc.datasets import CPTAC, select_peptide_subset
from ddmc.figures.common import getSetup, plot_motifs


def makeFigure():
    # Increase number of peptides and components for actual figure
    p_signal = CPTAC().get_p_signal()
    model = DDMC(n_components=16, seq_weight=100).fit(p_signal)

    ax, f = getSetup((9, 9), (4, 4))
    clusters, pssms = model.get_pssms(PsP_background=False)
    ylabels = np.arange(0, 21, 5)
    xlabels = [20, 21, 22, 23, 24, 25]
    for cluster in clusters:
        cluster_label = "Cluster " + str(cluster)
        plot_motifs(pssms[cluster], ax=ax[cluster], titles=cluster_label, yaxis=[0, 10])
        if cluster not in ylabels:
            ax[cluster].set_ylabel("")
            ax[cluster].get_yaxis().set_visible(False)
        if cluster not in xlabels:
            ax[cluster].set_xlabel("")
            ax[cluster].get_xaxis().set_visible(False)

    return f
