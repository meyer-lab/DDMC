"""
This creates Supplemental Figure 4: Island effect across radiis and time points 
"""
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from ..distances import PlotRipleysK, BarPlotRipleysK_TimePlots


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (6, 5))

    # Add subplot labels
    # subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    mutants = ['PC9', 'KO', 'KIN', 'KD', 'M4', 'M5', 'M7', 'M10', 'M11', 'M15']
    lines = ["WT", "KO", "KI", "KD", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F", "Y821F"] 


    for ii, mut in enumerate(mutants):
        PlotRipleysK('48hrs', mut, ['ut', 'e', 'ae'], 6, ax=ax[ii], title=lines[ii])

    # WT time points
    folder = 'PC9_TimeCourse'
    extensions = ['C1', 'F1', 'D1']
    radius = np.linspace(1.5, 14.67, 1)
    treatments = ['UT', "Erl", "AF154/Erl"]
    mutant = 'PC9 WT'
    l = [0, 5, 10, 15]
    for jj in range(20):
        extensions = ['C1_' + str(jj*3), 'F1_' + str(jj*3), 'D1_' + str(jj*3)]
        BarPlotRipleysK_TimePlots(folder, mutant, extensions, treatments, radius, ax[jj + 10])
        ax[jj + 10].set_title(str(jj*3) + ' hrs')
        if jj != 0:
            ax[jj + 10].get_legend().remove()
        if jj not in l:
            ax[jj + 10].set_ylabel("")

    return f