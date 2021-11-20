"""
This creates Supplemental Figure 7: Adjusted Mutual Information across clustering methods 
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.cluster import KMeans, AffinityPropagation, Birch, SpectralClustering, MeanShift, AgglomerativeClustering
from msresist.clustering import MassSpecClustering
from .common import subplotLabel, getSetup
from ..pre_processing import filter_NaNpeptides
from sklearn.metrics import adjusted_mutual_info_score


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))

    # Set plotting format
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    sns.set(style="whitegrid", font_scale=1, color_codes=True, palette="colorblind", rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6})

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
               SpectralClustering(n_clusters=15, affinity="nearest_neighbors"),
               AgglomerativeClustering(n_clusters=15),
               AgglomerativeClustering(n_clusters=15, linkage="average"),
               AgglomerativeClustering(n_clusters=15, linkage="complete")]

    labelsOut = np.empty((d.shape[1], len(methods) + 1), dtype=int)

    for ii, m in enumerate(methods):
        m.fit(d.T)
        labelsOut[:, ii] = m.labels_

    labelsOut[:, -1] = MassSpecClustering(i, ncl=30, SeqWeight=100, distance_method="Binomial", random_state=5).fit(d).predict()

    # How many clusters do we get in each instance
    print(np.amax(labelsOut, axis=0))

    mutInfo = np.zeros((labelsOut.shape[1], labelsOut.shape[1]), dtype=float)
    for ii in range(mutInfo.shape[0]):
        for jj in range(ii + 1):
            mutInfo[ii, jj] = adjusted_mutual_info_score(labelsOut[:, ii], labelsOut[:, jj])
            mutInfo[jj, ii] = mutInfo[ii, jj]

    sns.heatmap(mutInfo, ax=ax[0])
    labels = ["Birch", "MeanShift", "k-means", "Affinity Propagation", "SpectralClustering", "Agglomerative Clustering—Ward", "Agglomerative Clustering—Average", "Agglomerative Clustering—Commplete", "DDMC"]
    ax[0].set_xticklabels(labels, rotation=90)
    ax[0].set_yticklabels(labels, rotation=0)

    return f
