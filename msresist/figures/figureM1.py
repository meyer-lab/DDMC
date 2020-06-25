"""
This creates Figure M1.
"""

from .common import subplotLabel, getSetup
import pandas as pd
from msresist.figures.figure3 import plotclusteraverages
from msresist.clustering import MassSpecClustering


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (1, 1))
    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv")

    d = X.select_dtypes(include=["float64"]).T
    i = X.select_dtypes(include=["object"])

    dred = d.iloc[:, :2000]
    ired = i.iloc[:2000, :]

    MSC = MassSpecClustering(ired, ncl=2, SeqWeight=2000, gmm_method="pom", distance_method="PAM250", n_runs=1).fit(dred, None)

    plotclusteraverages(ax[0], MSC.transform(dred).T, dred.index)

    # Add subplot labels
    subplotLabel(ax)

    return f
