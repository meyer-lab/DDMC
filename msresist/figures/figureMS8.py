"""
This creates Supplemental Figure 7: Predicting STK11 genotype using different clustering strategies.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, AffinityPropagation, Birch, SpectralClustering, MeanShift, AgglomerativeClustering
from msresist.clustering import MassSpecClustering
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from sklearn.metrics import adjusted_mutual_info_score


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    # Signaling
    X = filter_NaNpeptides(pd.read_csv("msresist/data/MS/CPTAC/CPTAC-preprocessedMotfis.csv").iloc[:, 1:], cut=1)

    # Fit DDMC to complete data
    d = np.array(X.select_dtypes(include=['float64']).T)
    i = X.select_dtypes(include=['object'])

    assert np.all(np.isfinite(d))

    methods = [Birch(n_clusters=15),
               MeanShift(),
               KMeans(n_clusters=15),
               AffinityPropagation(),
               # OPTICS(), No clusters
               # DBSCAN(), No clusters
               SpectralClustering(n_clusters=15, affinity="nearest_neighbors"),
               AgglomerativeClustering(n_clusters=15),
               AgglomerativeClustering(n_clusters=15, linkage="average"),
               AgglomerativeClustering(n_clusters=15, linkage="complete")]

    labelsOut = np.empty((d.shape[1], len(methods) + 3), dtype=int)

    for ii, m in enumerate(methods):
        m.fit(d.T)
        labelsOut[:, ii] = m.labels_

    labelsOut[:, -3] = MassSpecClustering(i, ncl=20, SeqWeight=0, distance_method="PAM250").fit(d).predict()
    labelsOut[:, -2] = MassSpecClustering(i, ncl=20, SeqWeight=300, distance_method="Binomial").fit(d).predict()
    labelsOut[:, -1] = MassSpecClustering(i, ncl=20, SeqWeight=10, distance_method="PAM250").fit(d).predict()

    # How many clusters do we get in each instance
    print(np.amax(labelsOut, axis=0))

    mutInfo = np.zeros((labelsOut.shape[1], labelsOut.shape[1]), dtype=float)
    for ii in range(mutInfo.shape[0]):
        for jj in range(ii + 1):
            mutInfo[ii, jj] = adjusted_mutual_info_score(labelsOut[:, ii], labelsOut[:, jj])
            mutInfo[jj, ii] = mutInfo[ii, jj]

    sns.heatmap(mutInfo, ax=ax[0])

    return f
