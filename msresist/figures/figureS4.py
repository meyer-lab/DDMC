
import pickle
import seaborn as sns
from .common import subplotLabel, getSetup
from msresist.figures.figure2 import plotCenters, plotMotifs


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((12, 5), (2, 5))

    # Add subplot labels
    subplotLabel(ax)

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    # Load DDMC
    with open("msresist/data/pickled_models/AXLmodel_PAM250_W2-5_5CL", "rb") as m:
        model = pickle.load(m)
    centers = model.transform()
    lines = ["WT", "KO", "KD", "KI", "Y634F", "Y643F", "Y698F", "Y726F", "Y750F ", "Y821F"]

    # Centers
    plotCenters(ax[:5], model, lines)

    # Plot motifs
    pssms = model.pssms(PsP_background=True)
    plotMotifs([pssms[0], pssms[1], pssms[2], pssms[3], pssms[4]], axes=ax[5:10], titles=["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"], yaxis=[0, 11])

    return f
