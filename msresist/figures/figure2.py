"""
This creates Figure 2.
"""

from .common import subplotLabel, getSetup
import pandas as pd
from msresist.pre_processing import preprocessing
from msresist.figures.figure1 import IndividualTimeCourses, barplot_UtErlAF154

pd.set_option("display.max_columns", 30)
endpointcolors = ["light grey", "dark grey", "light navy blue", "jungle green"]


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 9), (2, 3), multz={8: 1, 10: 1})

    # # Heatmap Signaling
    # ax[6].axis("off")

    # # Read in Mass Spec data
    # X = preprocessing(Axlmuts_ErlAF154=True, Vfilter=True, FCfilter=True, log2T=True, mc_row=True)
    # d = X.select_dtypes(include=['float64']).T
    # d.index = all_lines

    # # Specific p-sites

    # erk = {"MAPK1":"Y187-p", "MAPK3":"Y204-p"}
    # erk_rn = ["ERK2", "ERK1"]

    # plot_AllSites(ax[7], X.copy(), "AXL", "AXL")
    # ax[7].legend(loc='upper left', prop={'size':8})
    # plot_AllSites(ax[8], X.copy(), "EGFR", "EGFR")
    # plot_IdSites(ax[9], X.copy(), erk, "ERK1/2", rn=erk_rn)

    return f
