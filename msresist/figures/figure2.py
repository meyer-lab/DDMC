"""
This creates Figure 2.
"""

from .common import subplotLabel, getSetup
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from msresist.pre_processing import preprocessing
from msresist.figures.figure1 import plotClustergram, selectpeptides, FC_timecourse, barplot_UtErlAF154, barplotFC_TvsUT
import matplotlib.image as mpimg

pd.set_option('display.max_columns', 30)
endpointcolors = ["light grey", "dark grey", "light navy blue", "jungle green"]


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 10), (3, 4), multz={8: 1, 10: 1})

    # blank out first axis for cartoon

    # ax[0].axis('off')

    # Read in Cell Migration data on collagen
    lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]
    glines = lines[2:]
    rwd = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/03062020-AXLmuts_EMT_RWD_Collagen_BR1.csv")
    ww = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/03062020-AXLmuts_EMT_WW_Collagen_BR1.csv")
    wc = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/03062020-AXLmuts_EMT_WC_Collagen_BR1.csv")
    rwdg = pd.read_csv("msresist/data/Phenotypic_data/AXLmutants/EMT/03062020-AXLmuts_EMT_GreenRWD_Collagen_BR1.csv")

    itp = 0
    ftp = 48
    t1 = "A/E"

    FC_timecourse(ax[0], rwd, itp, ftp, lines, t1, "Cell Migration - Erl + AF154", "Relative Wound Density")
    FC_timecourse(ax[2], rwdg, itp, ftp, glines, t1, "Cell Migration - Erl + AF154", "Relative Wound Density (AXL+)")
    FC_timecourse(ax[4], ww, itp, ftp, lines, t1, "Cell Migration - Erl + AF154", "Wound Width")
    FC_timecourse(ax[6], wc, itp, ftp, lines, t1, "Cell Migration - Erl + AF154", "Wound Confluency")

    ftp = 24
    tr1 = ['UT', 'AF', ' E', 'A/E']
    tr2 = ['UT', 'AF154', 'Erlotinib', 'Erl + AF154']

    c = ["white", "greyish blue", "deep blue"]
    barplotFC_TvsUT(ax[1], rwd, itp, ftp, lines, tr1, tr2, "Relative Wound Density - 24h", colors=c)
    c = ["white", "greeny grey", "forest green"]
    barplotFC_TvsUT(ax[3], rwdg, itp, ftp, glines, tr1, tr2, "Relative Wound Density (AXL+) - 24h", colors=c)
    c = ["white", "light grey", "slate grey", "green brown"]
    barplot_UtErlAF154(ax[5], lines, ww, itp, ftp, tr1, tr2, "Wound Width - 24h", " ", colors=c)
    barplot_UtErlAF154(ax[7], lines, wc, itp, ftp, tr1, tr2, "Wound Confluency - 24h", " ", colors=c)

    hm_af154 = mpimg.imread('msresist/data/Signaling/CM_reducedHM_AF154.png')
    hm_erl = mpimg.imread('msresist/data/Signaling/CM_reducedHM_Erl.png')
    ax[8].imshow(hm_af154)
    ax[8].axis("off")
    ax[9].imshow(hm_erl)
    ax[9].axis("off")

    # Add subplot labels
    subplotLabel(ax)

    return f
