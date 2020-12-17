"""Explore maximum number of clusters"""

import numpy as np
import pandas as  pd
import seaborn as sns
from .common import subplotLabel, getSetup
from ..clustering import MassSpecClustering
from ..figures.figure3 import plotUpstreamKinases


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((5, 5), (1, 2))

    # Set plotting format
    sns.set(style="whitegrid", font_scale=1.2, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

    X = pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:]
    d = X.select_dtypes(include=["float64"]).T
    i = X.select_dtypes(include=['object'])

    model = MassSpecClustering(i, ncl=30, SeqWeight=15, distance_method="Binomial").fit(d, "NA")
    plotUpstreamKinases(model, ax=ax[:2], clusters_=[11, 12], n_components=4, pX=1)

    return f