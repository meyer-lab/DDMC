import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    Birch,
    SpectralClustering,
    MeanShift,
    AgglomerativeClustering,
)
from ddmc.clustering import DDMC
from ddmc.datasets import CPTAC, filter_incomplete_peptides
from ddmc.figures.common import getSetup
from sklearn.metrics import adjusted_mutual_info_score


def makeFigure():
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1), labels=False)

    p_signal = filter_incomplete_peptides(
        CPTAC().get_p_signal(), sample_presence_ratio=1
    )

    methods = [
        Birch(n_clusters=15),
        MeanShift(),
        KMeans(n_clusters=15),
        AffinityPropagation(),
        SpectralClustering(n_clusters=15, affinity="nearest_neighbors"),
        AgglomerativeClustering(n_clusters=15),
        AgglomerativeClustering(n_clusters=15, linkage="average"),
        AgglomerativeClustering(n_clusters=15, linkage="complete"),
    ]

    labels_by_method = np.empty((p_signal.shape[0], len(methods) + 1), dtype=int)

    for ii, m in enumerate(methods):
        m.fit(p_signal.values)
        labels_by_method[:, ii] = m.labels_

    labels_by_method[:, -1] = (
        DDMC(
            n_components=30,
            seq_weight=100,
            distance_method="Binomial",
            random_state=5,
        )
        .fit(p_signal)
        .predict()
    )

    mutInfo = np.zeros(
        (labels_by_method.shape[1], labels_by_method.shape[1]), dtype=float
    )
    for ii in range(mutInfo.shape[0]):
        for jj in range(ii + 1):
            mutInfo[ii, jj] = adjusted_mutual_info_score(
                labels_by_method[:, ii], labels_by_method[:, jj]
            )
            mutInfo[jj, ii] = mutInfo[ii, jj]

    sns.heatmap(mutInfo, ax=ax[0])
    labels = [
        "Birch",
        "MeanShift",
        "k-means",
        "Affinity Propagation",
        "SpectralClustering",
        "Agglomerative Clustering—Ward",
        "Agglomerative Clustering—Average",
        "Agglomerative Clustering—Commplete",
        "DDMC",
    ]
    ax[0].set_xticklabels(labels, rotation=90)
    ax[0].set_yticklabels(labels, rotation=0)

    return f
