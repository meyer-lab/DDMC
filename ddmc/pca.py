""" PCA functions """
from typing import List

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA


def pca_dfs(scores, loadings, df, n_components, sIDX, lIDX):
    """build PCA scores and loadings data frames."""
    dScor = pd.DataFrame()
    dLoad = pd.DataFrame()
    for i in range(n_components):
        cpca = "PC" + str(i + 1)
        dScor[cpca] = scores[:, i]
        dLoad[cpca] = loadings[i, :]

    for j in sIDX:
        dScor[j] = list(df[j])
    # populate the "Cluster" col with the names of the clusters from df
    dLoad[lIDX] = df.select_dtypes(include=["float64"]).columns
    return dScor, dLoad


def plotPCA(
    axes: List,
    cluster_centers: pd.DataFrame,
    n_components: int,
    scores_ind,
    loadings_ind,
    hue_scores=None,
    style_scores=None,
    pvals=None,
    style_load=None,
    legendOut=False,
    quadrants=True,
):
    """Plot PCA scores and loadings."""
    pca = PCA(n_components=n_components)
    dScor_ = pca.fit_transform(cluster_centers.select_dtypes(include=["float64"]))
    dLoad_ = pca.components_
    dScor_, dLoad_ = pca_dfs(dScor_, dLoad_, cluster_centers, n_components, scores_ind, loadings_ind)
    varExp = np.round(pca.explained_variance_ratio_, 2)

    # Scores
    sns.scatterplot(
        x="PC1",
        y="PC2",
        data=dScor_,
        hue=hue_scores,
        style=style_scores,
        ax=axes[0],
        **{"linewidth": 0.5, "edgecolor": "k"}
    )
    axes[0].set_title("PCA Scores")
    axes[0].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    axes[0].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    axes[0].legend(prop={"size": 8})
    if legendOut:
        axes[0].legend(
            bbox_to_anchor=(1.05, 1),
            loc=2,
            borderaxespad=0,
            labelspacing=0.2,
            prop={"size": 8},
        )

    # Loadings
    if isinstance(pvals, np.ndarray):
        dLoad_["p-value"] = pvals
        sns.scatterplot(
            x="PC1",
            y="PC2",
            data=dLoad_,
            hue="p-value",
            style=style_load,
            ax=axes[1],
            **{"linewidth": 0.5, "edgecolor": "k"}
        )
    else:
        sns.scatterplot(
            x="PC1",
            y="PC2",
            data=dLoad_,
            style=style_load,
            ax=axes[1],
            **{"linewidth": 0.5, "edgecolor": "k"}
        )

    axes[1].set_title("PCA Loadings")
    axes[1].set_xlabel("PC1 (" + str(int(varExp[0] * 100)) + "%)", fontsize=10)
    axes[1].set_ylabel("PC2 (" + str(int(varExp[1] * 100)) + "%)", fontsize=10)
    axes[1].legend(prop={"size": 8})
    for j, txt in enumerate(dLoad_[loadings_ind]):
        axes[1].annotate(
            txt, (dLoad_["PC1"][j] + 0.001, dLoad_["PC2"][j] + 0.001), fontsize=10
        )

    if quadrants:
        axes[0].axhline(0, ls="--", color="lightgrey")
        axes[0].axvline(0, ls="--", color="lightgrey")
        axes[1].axhline(0, ls="--", color="lightgrey")
        axes[1].axvline(0, ls="--", color="lightgrey")
